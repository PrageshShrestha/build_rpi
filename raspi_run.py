#!/usr/bin/env python3
"""
raspi_run.py — Air Unit — Raspberry Pi 5
Reads test.mp4 → YOLO inference → JPEG → raw 802.11 → air
Run: sudo ~/yolo_env/bin/python3 raspi_run.py
"""

import socket
import struct
import time
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ── CONFIG ── change only these ──────────────
VIDEO_FILE   = "/home/pi/test.mp4"
YOLO_MODEL   = "/home/pi/best2.pt"
WIFI_CARD    = "wlan1"          # ← change to your card from Step 7
CHANNEL      = 6
MAGIC        = b'DRONE001'
CHUNK_SIZE   = 1400
JPEG_QUALITY = 75
FRAME_W      = 640
FRAME_H      = 480
YOLO_IMGSZ   = 320
CONF         = 0.4
LOOP         = True
# ─────────────────────────────────────────────

RADIOTAP = bytes([
    0x00, 0x00, 0x0c, 0x00,
    0x04, 0x80, 0x00, 0x00,
    0x02,   # 1Mbps = max range
    0x00, 0x18, 0x00,
])
DOT11 = bytes([
    0x08, 0x01, 0x00, 0x00,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0x13,0x22,0x33,0x44,0x55,0x66,
    0x13,0x22,0x33,0x44,0x55,0x66,
    0x00,0x00,
])
PKT_HEADER = RADIOTAP + DOT11


def set_monitor_mode():
    import subprocess
    print(f"[*] Setting {WIFI_CARD} to monitor mode...")
    for cmd in [
        ["ip", "link", "set", WIFI_CARD, "down"],
        ["iw", WIFI_CARD, "set", "monitor", "none"],
        ["ip", "link", "set", WIFI_CARD, "up"],
        ["iw", WIFI_CARD, "set", "channel", str(CHANNEL)],
    ]:
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            print(f"[!] {' '.join(cmd)}: {r.stderr.decode().strip()}")
    print(f"[+] Monitor mode set on channel {CHANNEL}")


def send_frame(sock, seq, jpeg_bytes):
    chunks = [jpeg_bytes[i:i+CHUNK_SIZE]
              for i in range(0, len(jpeg_bytes), CHUNK_SIZE)]
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        header  = struct.pack(">LHHH", seq, idx, total, len(chunk))
        payload = MAGIC + header + chunk
        packet  = PKT_HEADER + payload
        try:
            sock.send(packet)
        except OSError as e:
            print(f"[!] Send error: {e}")
        time.sleep(0.0001)


def main():
    set_monitor_mode()

    print(f"[*] Opening raw socket on {WIFI_CARD}...")
    try:
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,
                             socket.htons(0x0800))
        sock.bind((WIFI_CARD, 0))
    except PermissionError:
        print("[!] Run with sudo!")
        sys.exit(1)
    except OSError as e:
        print(f"[!] Socket error: {e}")
        sys.exit(1)
    print("[+] Socket ready")

    print(f"[*] Loading YOLO: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    print("[+] YOLO loaded")

    print(f"[*] Opening video: {VIDEO_FILE}")
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"[!] Cannot open {VIDEO_FILE}")
        sys.exit(1)
    print("[+] Video opened")

    seq      = 0
    frame_n  = 0
    t_start  = time.time()
    print("\n[*] Transmitting... Ctrl+C to stop\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if LOOP:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[*] Video ended.")
                    break

            frame_n += 1
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # YOLO
            results   = model(frame, imgsz=YOLO_IMGSZ,
                              conf=CONF, verbose=False)
            annotated = results[0].plot()

            # stats overlay
            elapsed = time.time() - t_start
            fps     = frame_n / elapsed if elapsed > 0 else 0
            n_det   = len(results[0].boxes)
            cv2.putText(annotated, f"FPS:{fps:.1f} DET:{n_det} SEQ:{seq}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,0), 2)

            # compress
            ok, buf = cv2.imencode('.jpg', annotated,
                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue

            # send
            send_frame(sock, seq, buf.tobytes())

            if frame_n % 30 == 0:
                print(f"[TX] seq={seq:05d} | frame={frame_n} | "
                      f"fps={fps:.1f} | dets={n_det} | "
                      f"size={len(buf)/1024:.1f}KB")
            seq += 1

    except KeyboardInterrupt:
        print("\n[*] Stopped")
    finally:
        cap.release()
        sock.close()
        elapsed = time.time() - t_start
        print(f"[+] Done — {seq} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
