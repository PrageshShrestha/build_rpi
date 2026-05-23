#!/usr/bin/env python3
"""
air_send.py — Run on Raspberry Pi (Air Unit)
─────────────────────────────────────────────
What this does:
  1. Reads a video file (or live camera)
  2. Runs YOLOv8 inference on every frame
  3. Draws bounding boxes on the frame
  4. Compresses to JPEG
  5. Splits into chunks
  6. Sends as raw 802.11 packets over monitor mode WiFi

Requirements:
  pip install ultralytics opencv-python numpy

Run:
  sudo python3 air_send.py

NOTE: Must run as root (sudo) for raw socket access
"""

import socket
import struct
import time
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG — change these to match your setup
# ─────────────────────────────────────────────

VIDEO_FILE   = "test_video.mp4"   # path to your test video on RPi
                                   # set to 0 for live camera

WIFI_CARD    = "wlan1"            # your external USB WiFi card
                                   # NOT wlan0 (that's built-in)
                                   # check with: ip link show

CHANNEL      = 6                  # WiFi channel (must match receiver)

MAGIC        = b'DRONE001'        # 8 byte identifier (must match receiver)

CHUNK_SIZE   = 1400               # bytes per packet (safe max for WiFi)

JPEG_QUALITY = 75                 # 0-100, lower = smaller = more range

FRAME_WIDTH  = 640                # resize frame before sending
FRAME_HEIGHT = 480

YOLO_MODEL   = "yolov8n.pt"      # nano = fastest on RPi
                                   # yolov8s.pt = slightly better accuracy

YOLO_IMGSZ   = 320               # inference size, smaller = faster

CONF_THRESH  = 0.4               # detection confidence threshold

LOOP_VIDEO   = True              # loop video file when it ends

# ─────────────────────────────────────────────
#  RAW 802.11 PACKET HEADERS
# ─────────────────────────────────────────────

# Radiotap header — tells WiFi driver how to transmit
# byte[8] = 0x02 means 1Mbps = maximum range
RADIOTAP_HEADER = bytes([
    0x00, 0x00,              # radiotap version: 0
    0x0c, 0x00,              # header length: 12 bytes
    0x04, 0x80, 0x00, 0x00,  # present flags: rate + tx flags
    0x02,                    # rate: 1Mbps (best range)
                             # use 0x04 for 2Mbps, 0x0B for 5.5Mbps
    0x00,                    # padding
    0x18, 0x00,              # TX flags
])

# 802.11 data frame header
DOT11_HEADER = bytes([
    0x08, 0x01,              # frame control: data frame
    0x00, 0x00,              # duration
    # destination MAC: broadcast (everyone receives it)
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    # source MAC: fake address (we're not a real WiFi device)
    0x13, 0x22, 0x33, 0x44, 0x55, 0x66,
    # BSSID: fake
    0x13, 0x22, 0x33, 0x44, 0x55, 0x66,
    0x00, 0x00,              # sequence control
])

# full static header prepended to every packet
PACKET_HEADER = RADIOTAP_HEADER + DOT11_HEADER

# ─────────────────────────────────────────────
#  CHUNK HEADER FORMAT
#  seq      = 4 bytes (frame sequence number)
#  idx      = 2 bytes (chunk index within frame)
#  total    = 2 bytes (total chunks in this frame)
#  size     = 2 bytes (size of this chunk's data)
#  ─────────────────────────────────
#  total chunk header = 10 bytes
# ─────────────────────────────────────────────
CHUNK_HEADER_FMT  = ">LHHH"
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FMT)  # = 10


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


def build_raw_packet(seq, chunk_idx, total_chunks, chunk_data):
    """Build a complete raw 802.11 packet for one chunk."""
    chunk_header = struct.pack(
        CHUNK_HEADER_FMT,
        seq,
        chunk_idx,
        total_chunks,
        len(chunk_data)
    )
    payload = MAGIC + chunk_header + chunk_data
    return PACKET_HEADER + payload


def send_frame(sock, seq, jpeg_bytes):
    """Split jpeg bytes into chunks and send each as a raw packet."""
    chunks = [
        jpeg_bytes[i:i + CHUNK_SIZE]
        for i in range(0, len(jpeg_bytes), CHUNK_SIZE)
    ]
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        packet = build_raw_packet(seq, idx, total, chunk)
        try:
            sock.send(packet)
        except OSError as e:
            print(f"[!] Send error on chunk {idx}/{total}: {e}")
        # tiny sleep to avoid overwhelming the buffer
        time.sleep(0.0001)


def main():
    # ── Setup monitor mode ──────────────────────
    set_monitor_mode(WIFI_CARD, CHANNEL)

    # ── Open raw socket ─────────────────────────
    print(f"[*] Opening raw socket on {WIFI_CARD}...")
    try:
        sock = socket.socket(
            socket.AF_PACKET,
            socket.SOCK_RAW,
            socket.htons(0x0800)
        )
        sock.bind((WIFI_CARD, 0))
    except PermissionError:
        print("[!] Permission denied — run with sudo!")
        sys.exit(1)
    except OSError as e:
        print(f"[!] Socket error: {e}")
        print(f"[!] Is {WIFI_CARD} available? Run: ip link show")
        sys.exit(1)

    print(f"[+] Raw socket ready")

    # ── Load YOLO model ─────────────────────────
    print(f"[*] Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    print(f"[+] YOLO loaded")

    # ── Open video source ───────────────────────
    source = VIDEO_FILE if VIDEO_FILE != "0" else 0
    print(f"[*] Opening video: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[!] Cannot open video: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"[+] Video opened — {total_frames} frames @ {fps_src:.1f} FPS")

    # ── Main loop ───────────────────────────────
    seq        = 0
    frame_num  = 0
    t_start    = time.time()

    print(f"\n[*] Transmitting... (Ctrl+C to stop)\n")

    try:
        while True:
            ret, frame = cap.read()

            # loop video if enabled
            if not ret:
                if LOOP_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[*] Video ended.")
                    break

            frame_num += 1

            # ── Resize ──────────────────────────
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ── YOLO Inference ──────────────────
            results   = model(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=CONF_THRESH,
                verbose=False     # suppress per-frame prints
            )
            annotated = results[0].plot()  # frame with boxes + labels

            # ── Add overlay info ────────────────
            elapsed = time.time() - t_start
            fps     = frame_num / elapsed if elapsed > 0 else 0
            n_det   = len(results[0].boxes)

            cv2.putText(annotated, f"FPS: {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Frame: {frame_num}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Detections: {n_det}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"SEQ: {seq}",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ── JPEG Compress ───────────────────
            ret_enc, jpeg_buf = cv2.imencode(
                '.jpg', annotated,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ret_enc:
                print(f"[!] Failed to encode frame {frame_num}")
                continue

            jpeg_bytes = jpeg_buf.tobytes()

            # ── Send over raw socket ─────────────
            send_frame(sock, seq, jpeg_bytes)

            # ── Print stats every 30 frames ──────
            if frame_num % 30 == 0:
                kb = len(jpeg_bytes) / 1024
                print(f"[TX] seq={seq:05d} | frame={frame_num} | "
                      f"size={kb:.1f}KB | fps={fps:.1f} | dets={n_det}")

            seq += 1

    except KeyboardInterrupt:
        print("\n[*] Stopped by user")

    finally:
        cap.release()
        sock.close()
        elapsed = time.time() - t_start
        print(f"\n[+] Done — sent {seq} frames in {elapsed:.1f}s "
              f"({seq/elapsed:.1f} fps avg)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        VIDEO_FILE = sys.argv[1]
    main()
