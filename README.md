# build_rpi

A real-time video streaming and object detection system for Raspberry Pi using raw 802.11 wireless transmission and YOLO inference.

## Overview

This project implements an air-to-ground video transmission pipeline optimized for embedded systems. The system captures video frames on a Raspberry Pi 5, performs YOLO-based object detection, compresses frames as JPEG, and transmits them over raw 802.11 wireless packets to a ground station running on Linux.

The implementation uses low-level socket programming to bypass standard network stack limitations, enabling direct control over packet structure and transmission timing.

## System Architecture

### Components

1. **Air Unit (Raspberry Pi 5)** - `raspi_run.py`
   - Reads video input from file or camera
   - Performs real-time object detection using YOLO
   - Encodes annotated frames as JPEG
   - Segments JPEG data into wireless packets
   - Transmits packets over raw 802.11

2. **Ground Unit (Linux Desktop/Laptop)** - `linux_run.py`
   - Operates WiFi adapter in monitor mode
   - Captures all raw 802.11 packets
   - Filters packets by magic bytes identifier
   - Reassembles fragmented JPEG data
   - Decodes frames and displays with statistics

### Packet Format

```
[Radiotap Header (12 bytes)][802.11 Header (24 bytes)][Magic (8 bytes)]
[Sequence (4)][Index (2)][Total (2)][Size (2)][JPEG Chunk Data]
```

- **Radiotap Header**: Hardware-provided frame metadata (added by WiFi driver)
- **802.11 Header**: Layer 2 frame control information
- **Magic Bytes**: Packet identification (default: `DRONE001`)
- **Chunk Header**: Frame sequence number, chunk index, total chunks, data size
- **Payload**: JPEG chunk data

## Requirements

### Air Unit (Raspberry Pi 5)

```
python3.11
opencv-python
numpy
ultralytics (YOLOv8)
```

### Ground Unit (Linux)

```
python3
opencv-python
numpy
iw (wireless configuration tool)
monitor-mode capable WiFi adapter
```

## Installation

### Raspberry Pi Setup

1. Create Python virtual environment:
   ```bash
   python3 -m venv ~/yolo_env
   source ~/yolo_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy ultralytics
   ```

3. Place video file and YOLO model:
   ```bash
   # Copy test.mp4 and best2.pt to home directory
   cp test.mp4 ~/test.mp4
   cp best2.pt ~/best2.pt
   ```

4. Update WiFi card name in `raspi_run.py`:
   ```python
   WIFI_CARD = "wlan1"  # Replace with your adapter
   ```

### Linux Ground Station Setup

1. Install system dependencies:
   ```bash
   sudo apt-get install iw
   ```

2. Install Python dependencies:
   ```bash
   pip install opencv-python numpy
   ```

3. Update WiFi card configuration in `linux_run.py`:
   ```python
   WIFI_CARD = "wlan1"  # Replace with your adapter
   CHANNEL = 6          # Must match Raspberry Pi setting
   ```

## Configuration Parameters

### raspi_run.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO_FILE` | `/home/pi/test.mp4` | Input video file path |
| `YOLO_MODEL` | `/home/pi/best2.pt` | YOLO model weights file |
| `WIFI_CARD` | `wlan1` | WiFi adapter interface name |
| `CHANNEL` | `6` | WiFi channel (must match ground unit) |
| `MAGIC` | `DRONE001` | Packet identifier (must match ground unit) |
| `CHUNK_SIZE` | `1400` | Bytes per wireless packet |
| `JPEG_QUALITY` | `75` | JPEG compression quality (0-100) |
| `FRAME_W` | `640` | Frame width in pixels |
| `FRAME_H` | `480` | Frame height in pixels |
| `YOLO_IMGSZ` | `320` | YOLO model input size |
| `CONF` | `0.4` | Detection confidence threshold |
| `LOOP` | `True` | Loop video on end |

### linux_run.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WIFI_CARD` | `wlan1` | WiFi adapter interface name |
| `CHANNEL` | `6` | WiFi channel to monitor |
| `MAGIC` | `DRONE001` | Expected packet identifier |
| `WINDOW_NAME` | `Drone YOLO Feed` | Display window title |
| `MAX_BUFFER_AGE` | `30` | Stale buffer cleanup threshold |

## Usage

### Starting the Air Unit

```bash
cd ~/build_rpi
sudo ~/yolo_env/bin/python3 raspi_run.py
```

Output:
```
[*] Setting wlan1 to monitor mode...
[+] Monitor mode set on channel 6
[*] Opening raw socket on wlan1...
[+] Socket ready
[*] Loading YOLO: /home/pi/best2.pt
[+] YOLO loaded
[*] Opening video: /home/pi/test.mp4
[+] Video opened

[*] Transmitting... Ctrl+C to stop

[TX] seq=00000 | frame=30 | fps=30.0 | dets=2 | size=45.3KB
```

### Starting the Ground Unit

```bash
cd ~/build_rpi
sudo python3 linux_run.py
```

Output:
```
[*] Setting wlan1 to monitor mode on channel 6...
[+] wlan1 ready in monitor mode on channel 6
[*] Opening raw capture socket on wlan1...
[+] Listening for drone packets on channel 6...
[*] Press Q in the video window to quit

[RX] seq=00000 | fps=30.0 | size=45.3KB | dropped=0 | rssi=-45dBm | foreign_pkts=142
```

A window will display the live video feed with FPS, frame size, latency, and signal strength overlays.

## Performance Tuning

### Bandwidth Optimization

- Increase `JPEG_QUALITY` for better image fidelity (increases bandwidth)
- Decrease `JPEG_QUALITY` to reduce required bandwidth
- Adjust `CHUNK_SIZE` based on WiFi adapter MTU and interference tolerance
- Smaller chunks improve robustness in noisy environments

### Latency Optimization

- Reduce `FRAME_W` and `FRAME_H` to process frames faster
- Lower `YOLO_IMGSZ` for faster inference (trade detection accuracy)
- Increase `CHUNK_SIZE` to reduce packet overhead (increases per-packet latency)

### Detection Accuracy

- Lower `CONF` threshold to catch more objects (increases false positives)
- Use higher resolution `YOLO_IMGSZ` for better accuracy
- Select appropriate YOLO model size (nano, small, medium, large, xlarge)

## Troubleshooting

### Monitor Mode Setup Failed

**Error**: `[!] Warning: iw command failed`

**Solution**:
1. Verify WiFi adapter supports monitor mode: `iw list | grep monitor`
2. Ensure adapter is not in use: `sudo systemctl stop wpa_supplicant`
3. Check adapter name: `ip link show`

### Socket Permission Denied

**Error**: `[!] Permission denied — run with sudo!`

**Solution**: Both scripts require root privileges for raw socket access.

```bash
sudo ~/yolo_env/bin/python3 raspi_run.py
sudo python3 linux_run.py
```

### No Packets Received

**Diagnosis**:
1. Verify both units use same channel: compare `CHANNEL` values
2. Verify magic bytes match: check `MAGIC` in both scripts
3. Check adapter channel: `iw dev wlan1 link`
4. Monitor WiFi traffic: `sudo tcpdump -i wlan1 -X`

### YOLO Model Not Found

**Error**: `[!] FileNotFoundError: [Errno 2] No such file or directory: '/home/pi/best2.pt'`

**Solution**: Place YOLO model in specified path or update `YOLO_MODEL` parameter.

### Dropped Frames

**Cause**: Insufficient bandwidth or interference

**Solutions**:
1. Reduce `JPEG_QUALITY`
2. Lower frame resolution (`FRAME_W`, `FRAME_H`)
3. Change WiFi channel to less congested frequency
4. Reduce video source FPS

## Technical Notes

### Raw Socket Implementation

Both scripts use raw sockets (`AF_PACKET`) to construct and transmit complete 802.11 frames. This provides:

- Direct control over layer 2 headers
- Ability to bypass normal network routing
- Custom frame timing and pacing
- Monitor mode packet injection

### Radiotap Header

The Radiotap header is platform-specific and set by the wireless driver. The ground unit automatically parses it to extract signal strength information.

### Packet Reassembly

The ground unit maintains per-sequence buffers to handle out-of-order packet arrival. Incomplete frames are discarded after `MAX_BUFFER_AGE` packets to prevent memory exhaustion.

### RSSI Calculation

Signal strength is extracted from the radiotap header (field offset depends on driver implementation). RSSI values range from 0 to -120 dBm where -30 dBm is excellent and -90 dBm is poor.

## Platform Compatibility

- **Air Unit**: Raspberry Pi 5 (Linux ARM64)
- **Ground Unit**: Linux (x86_64, ARM64)
- **Not Supported**: macOS, Windows (AF_PACKET is Linux-specific)

## Future Enhancements

- FEC (Forward Error Correction) for packet loss resilience
- Adaptive bitrate control based on RSSI
- Multiple ground stations with packet aggregation
- Hardware acceleration for JPEG encoding/decoding
- IPv6/UDP tunneling fallback for non-monitor-mode networks

## License

See LICENSE file in repository.

## Author

Pragesh Shrestha
