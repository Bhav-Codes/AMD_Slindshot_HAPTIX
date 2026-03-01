"""Run this to see results — usage: python main.py [video_file]"""
import cv2
import numpy as np
import csv
import os
import sys
import json
import serial
import time
from bounce_detector import BounceDetector, map_to_grid

# ===============================
# Load calibration data
# ===============================
calib = {}
if os.path.exists("calibration_data.json"):
    with open("calibration_data.json") as f:
        calib = json.load(f)
    print("Loaded calibration_data.json")
elif not os.path.exists("table_corners.npy"):
    print("Please run caliberate_table.py first")
    exit()

# Table corners (prefer JSON, fallback to npy)
if "table_corners" in calib:
    table_corners = np.array(calib["table_corners"], dtype=np.float32)
else:
    table_corners = np.load("table_corners.npy").astype(np.float32)
print("Loaded table corners:", table_corners)

# Table color HSV (from calibration or default green)
table_hsv_lower = np.array(calib.get("table_hsv_lower", [35, 80, 80]))
table_hsv_upper = np.array(calib.get("table_hsv_upper", [85, 255, 255]))
print(f"Table HSV: {table_hsv_lower} → {table_hsv_upper}")

# Ball color (from calibration or default yellow)
ball_color = calib.get("ball_color", "yellow")
print(f"Ball color mode: {ball_color}")

if ball_color == "white":
    ball_hsv_lower = np.array([0, 0, 200])
    ball_hsv_upper = np.array([180, 60, 255])
else:  # yellow / orange
    ball_hsv_lower = np.array([15, 150, 150])
    ball_hsv_upper = np.array([35, 255, 255])

# ===============================
# Video input — CLI arg > calibration > fallback
# ===============================
if len(sys.argv) > 1:
    video_file = sys.argv[1]
elif "video_file" in calib:
    video_file = calib["video_file"]
else:
    video_file = "input.mp4"

if not os.path.exists(video_file):
    print(f"Error: Video file '{video_file}' not found")
    exit()

print(f"Opening video: {video_file}")

# ===============================
# TABLE CORNER LABELS
# ===============================
corner_labels = {
    0: "(0,0)",   # bottom-left
    1: "(3,0)",   # bottom-right
    2: "(3,2)",   # top-right
    3: "(0,2)"    # top-left
}

TABLE_WIDTH  = 1.525
TABLE_LENGTH = 2.74

table_pts = np.array([
    [0.0, 0.0],
    [TABLE_WIDTH, 0.0],
    [TABLE_WIDTH, TABLE_LENGTH],
    [0.0, TABLE_LENGTH]
], dtype=np.float32)

H, _ = cv2.findHomography(table_corners, table_pts)

# ===============================
# BOUNCE DETECTOR
# ===============================
detector = BounceDetector(buffer_size=3, cooldown_frames=5)
current_bounce = ("--", "--")
bounce_display_timer = 0
bounce_impact_pos = None

# ===============================
# SERIAL SETUP (Haptic Board)
# ===============================
SERIAL_PORT = 'COM7'  # CHANGE THIS to your ESP32 port
SERIAL_BAUD = 115200
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    time.sleep(1)
    print(f"Connected to haptic board on {SERIAL_PORT}")
except Exception as e:
    print(f"Could not connect to serial port {SERIAL_PORT}: {e}")
    print("Running in visualization-only mode.")

# ===============================
# Video capture
# ===============================
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

frame_id = 0

DISPLAY_SCALE = 1.5
HIT_DISPLAY_FRAMES = int(0.35 * fps)

# ===============================
# Video output (processed)
# ===============================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "processed_output.mp4",
    fourcc,
    fps,
    (frame.shape[1], frame.shape[0])
)

# ===============================
# CSV output
# ===============================
csv_file = open("ball_events.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "event", "grid_col", "grid_row", "table_x", "table_y"])

# ===============================
# Tracking state
# ===============================
last_seen_x = last_seen_y = None
last_table_x = last_table_y = None

# ===============================
# Main loop
# ===============================
while True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Ball mask — using calibrated color
    mask = cv2.inRange(hsv, ball_hsv_lower, ball_hsv_upper)

    # Table mask — subtract table color to avoid false detections
    table_mask = cv2.inRange(hsv, table_hsv_lower, table_hsv_upper)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(table_mask))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = cy = None
    for c in contours:
        area = cv2.contourArea(c)
        if 40 < area < 600:
            x, y, w, h = cv2.boundingRect(c)
            if 0.55 < w / float(h) < 1.8:
                cx = x + w // 2
                cy = y + h // 2
                break

    # ===============================
    # Ball + table coordinate
    # ===============================
    if cx is not None:
        last_seen_x, last_seen_y = cx, cy

        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        last_table_x, last_table_y = mapped

        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # ── Run bounce detector ──
        is_bounce = detector.update(cy, cx=cx, frame_id=frame_id)

        if is_bounce:
            MARGIN = 0.08

            impact_pos = detector.peak_position
            if impact_pos is not None:
                ipx, ipy = impact_pos
                ipt = np.array([[[ipx, ipy]]], dtype=np.float32)
                impact_mapped = cv2.perspectiveTransform(ipt, H)[0][0]
                imp_tx, imp_ty = impact_mapped
            else:
                ipx, ipy = cx, cy
                imp_tx, imp_ty = last_table_x, last_table_y

            on_table = (-MARGIN <= imp_tx <= TABLE_WIDTH + MARGIN and
                        -MARGIN <= imp_ty <= TABLE_LENGTH + MARGIN)
            if on_table:
                col, row = map_to_grid(imp_tx, imp_ty)
                current_bounce = (col, row)
                bounce_display_timer = HIT_DISPLAY_FRAMES
                bounce_impact_pos = (ipx, ipy)

                writer.writerow([
                    frame_id, "BOUNCE", col, row,
                    f"{imp_tx:.3f}", f"{imp_ty:.3f}"
                ])
                print(f"[BOUNCE] frame {frame_id} (peak@{detector.peak_frame_id})  "
                      f"grid=({col},{row})  "
                      f"table=({imp_tx:.2f}, {imp_ty:.2f})")

                # TRIGGER HAPTIC BOARD
                if ser and ser.is_open:
                    msg = f"{col},{row}\n"
                    ser.write(msg.encode())

    # ── Bounce highlight ring ──
    if bounce_display_timer > 0 and bounce_impact_pos is not None:
        cv2.circle(frame, bounce_impact_pos, 14, (255, 0, 0), 3)
    bounce_display_timer = max(0, bounce_display_timer - 1)

    # ===============================
    # Table corners
    # ===============================
    for idx, (x, y) in enumerate(table_corners.astype(int)):
        cv2.circle(frame, (x, y), 6, (0, 0, 0), -1)
        cv2.putText(
            frame,
            corner_labels[idx],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ===============================
    # DISPLAY TEXT
    # ===============================
    h, w, _ = frame.shape

    cv2.putText(
        frame,
        f"Frame: {frame_id}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 255),
        3
    )

    x_right = w - 300
    y_right = 60

    cv2.putText(
        frame,
        f"Bounce: ({current_bounce[0]}, {current_bounce[1]})",
        (x_right, y_right),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255,255,255),
        3
    )

    # Show detector phase
    phase_color = {
        "RISING":  (0, 255, 0),
        "FALLING": (0, 165, 255),
        "IMPACT":  (0, 0, 255),
    }
    phase_name = detector.phase.value
    cv2.putText(
        frame,
        f"Phase: {phase_name}",
        (x_right, y_right + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        phase_color.get(phase_name, (255, 255, 255)),
        2
    )

    # ===============================
    # Save processed frame
    # ===============================
    out.write(frame)

    # ===============================
    # Show
    # ===============================
    frame_disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Ball Tracking", frame_disp)
    cv2.setWindowProperty("Ball Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        break

    # ---- Read next frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

# ===============================
# Cleanup
# ===============================
if ser and ser.is_open:
    ser.close()
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
