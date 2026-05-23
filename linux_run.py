#!/usr/bin/env python3
"""
ground_receive.py — Run on Ground Linux Machine (Laptop/Desktop)
─────────────────────────────────────────────────────────────────
What this does:
  1. Sets WiFi card to monitor mode
  2. Captures ALL raw 802.11 packets flying through the air
  3. Filters only OUR drone packets (by magic bytes)
  4. Reassembles chunks back into complete frames
  5. Decodes JPEG → numpy array
  6. Displays live video with detections + stats

Requirements:
  pip install opencv-python numpy

Run:
  sudo python3 ground_receive.py

NOTE: Must run as root (sudo) for raw socket access
NOTE: AF_PACKET only works on Linux — not macOS/Windows
"""

import socket
import struct
import time
import sys
import cv2
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONFIG — must match air_send.py exactly!
# ─────────────────────────────────────────────

WIFI_CARD   = "wlan1"      # your external USB WiFi card in monitor mode
                            # must be same channel as air side

CHANNEL     = 6            # must match air_send.py

MAGIC       = b'DRONE001'  # must match air_send.py exactly

WINDOW_NAME = "Drone YOLO Feed"

# How many old incomplete frames to keep before discarding
# (higher = more memory, more tolerance for out-of-order packets)
MAX_BUFFER_AGE = 30

# ─────────────────────────────────────────────
#  PACKET STRUCTURE (must match air_send.py)
# ─────────────────────────────────────────────

MAGIC_LEN         = len(MAGIC)       # 8 bytes
CHUNK_HEADER_FMT  = ">LHHH"
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FMT)  # 10 bytes

# Offsets inside raw received packet:
#   [0   : 12 ] = radiotap header    (added by WiFi hardware)
#   [12  : 36 ] = 802.11 dot11 header
#   [36  : 44 ] = our MAGIC bytes
#   [44  : 54 ] = our chunk header (seq, idx, total, size)
#   [54  :    ] = actual jpeg chunk data

RADIOTAP_LEN  = 12
DOT11_LEN     = 24
PAYLOAD_START = RADIOTAP_LEN + DOT11_LEN   # = 36


def set_monitor_mode(card, channel):
    """Set WiFi card to monitor mode on given channel."""
    import subprocess
    print(f"[*] Setting {card} to monitor mode on channel {channel}...")
    cmds = [
        ["ip", "link", "set", card, "down"],
        ["iw", card, "set", "monitor", "none"],
        ["ip", "link", "set", card, "up"],
        ["iw", card, "set", "channel", str(channel)],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"[!] Warning: {' '.join(cmd)} failed: {result.stderr.decode()}")
    print(f"[+] {card} ready in monitor mode on channel {channel}")


def parse_packet(raw):
    """
    Parse a raw 802.11 packet.
    Returns (seq, idx, total, chunk_data) or None if not our packet.
    """
    # minimum length check
    min_len = PAYLOAD_START + MAGIC_LEN + CHUNK_HEADER_SIZE
    if len(raw) < min_len:
        return None

    # check magic bytes
    magic_start = PAYLOAD_START
    magic_end   = magic_start + MAGIC_LEN
    if raw[magic_start:magic_end] != MAGIC:
        return None

    # parse chunk header
    header_start = magic_end
    header_end   = header_start + CHUNK_HEADER_SIZE
    seq, idx, total, chunk_size = struct.unpack(
        CHUNK_HEADER_FMT,
        raw[header_start:header_end]
    )

    # extract chunk data
    data_start = header_end
    data_end   = data_start + chunk_size
    chunk_data = raw[data_start:data_end]

    if len(chunk_data) != chunk_size:
        return None   # truncated packet

    return seq, idx, total, chunk_data


def get_rssi(raw):
    """
    Try to read signal strength (RSSI) from radiotap header.
    Returns dBm value or None if not available.
    """
    try:
        # radiotap present flags at bytes 4-7
        present = struct.unpack("<L", raw[4:8])[0]
        # bit 5 = RSSI field present
        if present & (1 << 5):
            # RSSI is usually at byte 14 in standard radiotap
            rssi = struct.unpack("b", raw[14:15])[0]  # signed byte
            return rssi
    except:
        pass
    return None


def decode_frame(jpeg_bytes):
    """Convert JPEG bytes → numpy BGR image array."""
    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def main():
    # ── Setup monitor mode ──────────────────────
    set_monitor_mode(WIFI_CARD, CHANNEL)

    # ── Open raw socket ─────────────────────────
    print(f"[*] Opening raw capture socket on {WIFI_CARD}...")
    try:
        sock = socket.socket(
            socket.AF_PACKET,
            socket.SOCK_RAW,
            socket.htons(0x0003)   # ETH_P_ALL = capture everything
        )
        sock.bind((WIFI_CARD, 0))
        sock.settimeout(2.0)       # 2s timeout so we can check for quit
    except PermissionError:
        print("[!] Permission denied — run with sudo!")
        sys.exit(1)
    except OSError as e:
        print(f"[!] Socket error: {e}")
        print(f"[!] Is {WIFI_CARD} available? Run: ip link show")
        sys.exit(1)

    print(f"[+] Listening for drone packets on channel {CHANNEL}...")
    print(f"[*] Press Q in the video window to quit\n")

    # ── Buffers ─────────────────────────────────
    # { seq_number: { chunk_idx: chunk_bytes } }
    frame_buffers = defaultdict(dict)

    # ── Stats ────────────────────────────────────
    frames_shown    = 0
    frames_dropped  = 0
    packets_rx      = 0
    packets_foreign = 0
    last_seq        = -1
    t_start         = time.time()
    t_last_frame    = time.time()
    last_rssi       = None

    # ── Create display window ─────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    try:
        while True:
            # ── Receive raw packet ───────────────
            try:
                raw, _ = sock.recvfrom(65535)
            except socket.timeout:
                # check if window was closed
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            # ── Try to parse as our packet ───────
            result = parse_packet(raw)

            if result is None:
                packets_foreign += 1
                continue

            seq, idx, total, chunk_data = result
            packets_rx += 1

            # try to get signal strength
            rssi = get_rssi(raw)
            if rssi is not None:
                last_rssi = rssi

            # ── Store chunk ──────────────────────
            frame_buffers[seq][idx] = chunk_data

            # ── Check if frame complete ──────────
            if len(frame_buffers[seq]) == total:

                # reassemble chunks in order
                jpeg_bytes = b""
                complete   = True
                for i in range(total):
                    if i not in frame_buffers[seq]:
                        complete = False
                        break
                    jpeg_bytes += frame_buffers[seq][i]

                if complete:
                    # decode JPEG → image
                    frame = decode_frame(jpeg_bytes)

                    if frame is not None:
                        # ── Calculate stats ──────
                        elapsed     = time.time() - t_start
                        fps         = frames_shown / elapsed if elapsed > 0 else 0
                        now         = time.time()
                        frame_delay = (now - t_last_frame) * 1000  # ms
                        t_last_frame = now
                        kb_size     = len(jpeg_bytes) / 1024

                        # detect dropped frames
                        if last_seq >= 0 and seq != last_seq + 1:
                            dropped = seq - last_seq - 1
                            frames_dropped += dropped
                        last_seq = seq

                        # ── Add stats overlay ────
                        # semi-transparent black bar at bottom
                        h, w = frame.shape[:2]
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, h-90), (w, h), (0,0,0), -1)
                        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                        stats = [
                            f"FPS: {fps:.1f}",
                            f"Frame: {seq}  |  Size: {kb_size:.1f}KB",
                            f"Dropped: {frames_dropped}  |  Latency: {frame_delay:.0f}ms",
                            f"RSSI: {last_rssi} dBm" if last_rssi else "RSSI: --",
                        ]
                        for i, text in enumerate(stats):
                            cv2.putText(
                                frame, text,
                                (10, h - 70 + i*18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 0), 1, cv2.LINE_AA
                            )

                        # ── Show frame ───────────
                        cv2.imshow(WINDOW_NAME, frame)
                        frames_shown += 1

                        # print stats every 60 frames
                        if frames_shown % 60 == 0:
                            print(
                                f"[RX] seq={seq:05d} | "
                                f"fps={fps:.1f} | "
                                f"size={kb_size:.1f}KB | "
                                f"dropped={frames_dropped} | "
                                f"rssi={last_rssi}dBm | "
                                f"foreign_pkts={packets_foreign}"
                            )

                    else:
                        print(f"[!] Failed to decode frame seq={seq}")

                # cleanup this frame buffer
                del frame_buffers[seq]

            # ── Cleanup old incomplete buffers ───
            stale = [k for k in frame_buffers if k < seq - MAX_BUFFER_AGE]
            for k in stale:
                del frame_buffers[k]

            # ── Handle keypress ──────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break

    except KeyboardInterrupt:
        print("\n[*] Stopped by user")

    finally:
        sock.close()
        cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        print(f"\n── Final Stats ──────────────────────")
        print(f"  Runtime:        {elapsed:.1f}s")
        print(f"  Frames shown:   {frames_shown}")
        print(f"  Frames dropped: {frames_dropped}")
        print(f"  Avg FPS:        {frames_shown/elapsed:.1f}" if elapsed > 0 else "")
        print(f"  Our packets:    {packets_rx}")
        print(f"  Foreign pkts:   {packets_foreign}")
        print(f"────────────────────────────────────")


if __name__ == "__main__":
    main()
