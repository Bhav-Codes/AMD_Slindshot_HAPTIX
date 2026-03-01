"""Run this to calibrate table corners, table color, and ball color."""
import cv2
import numpy as np
import json
import sys
import os

# ── Video file (CLI arg or default) ──
video_path = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found")
    exit()

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video")
    exit()

# Keep a clean copy for color sampling
frame_clean = frame.copy()

points = []
stage = "corners"  # "corners" → "table_color"

def mouse_click(event, x, y, flags, param):
    global stage
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if stage == "corners" and len(points) < 4:
        points.append((x, y))
        print(f"Corner {len(points)}: ({x}, {y})")
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        if len(points) == 4:
            # Draw the quadrilateral
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            stage = "table_color"
            print("\n>> Now click ONCE inside the table surface (for color sampling)")
        cv2.imshow("Calibration", frame)

    elif stage == "table_color" and len(points) == 4:
        points.append((x, y))
        print(f"Table color sample point: ({x}, {y})")
        cv2.circle(frame, (x, y), 10, (255, 0, 255), 2)
        cv2.imshow("Calibration", frame)

cv2.imshow("Calibration", frame)
cv2.setMouseCallback("Calibration", mouse_click)

print("=" * 50)
print(f"Calibrating from: {video_path}")
print("=" * 50)
print("\nClick table corners in this order:")
print("  1. Bottom-Left")
print("  2. Bottom-Right")
print("  3. Top-Right")
print("  4. Top-Left")
print("  5. (after corners) Click INSIDE the table for color")
print("\nPress ESC when done.")

while True:
    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1) & 0xFF
    # Auto-close after 5th click or on ESC
    if key == 27 or len(points) >= 5:
        break

cv2.destroyAllWindows()

# ── Validate clicks ──
if len(points) < 4:
    print("ERROR: You must click at least 4 corner points")
    exit()

corners = points[:4]

# ── Sample table color from 5th click ──
if len(points) >= 5:
    sx, sy = points[4]
    # Sample a 21x21 patch around the click
    half = 10
    h_img, w_img = frame_clean.shape[:2]
    y1 = max(0, sy - half)
    y2 = min(h_img, sy + half + 1)
    x1 = max(0, sx - half)
    x2 = min(w_img, sx + half + 1)
    patch = frame_clean[y1:y2, x1:x2]
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h_vals = patch_hsv[:, :, 0].flatten()
    s_vals = patch_hsv[:, :, 1].flatten()
    v_vals = patch_hsv[:, :, 2].flatten()

    table_hsv_lower = [
        int(max(0, np.percentile(h_vals, 5) - 10)),
        int(max(0, np.percentile(s_vals, 5) - 40)),
        int(max(0, np.percentile(v_vals, 5) - 40))
    ]
    table_hsv_upper = [
        int(min(180, np.percentile(h_vals, 95) + 10)),
        int(min(255, np.percentile(s_vals, 95) + 40)),
        int(min(255, np.percentile(v_vals, 95) + 40))
    ]
    print(f"\nTable HSV range: {table_hsv_lower} → {table_hsv_upper}")
else:
    # Fallback to hardcoded green if no 5th click
    print("\nNo table color point clicked — using default green range")
    table_hsv_lower = [35, 80, 80]
    table_hsv_upper = [85, 255, 255]

# ── Ball color selection ──
print("\n" + "=" * 50)
print("Select ball color:")
print("  1. Yellow / Orange (default)")
print("  2. White")
print("=" * 50)
choice = input("Enter 1 or 2: ").strip()
ball_color = "white" if choice == "2" else "yellow"
print(f"Ball color set to: {ball_color}")

# ── Save everything ──
# 1. table_corners.npy (backward compat)
np.save("table_corners.npy", np.array(corners, dtype=np.float32))

# 2. calibration_data.json (full config)
calib_data = {
    "table_corners": corners,
    "table_hsv_lower": table_hsv_lower,
    "table_hsv_upper": table_hsv_upper,
    "ball_color": ball_color,
    "video_file": video_path
}
with open("calibration_data.json", "w") as f:
    json.dump(calib_data, f, indent=2)

print(f"\nSaved table_corners.npy")
print(f"Saved calibration_data.json")
print(f"  Table HSV: {table_hsv_lower} → {table_hsv_upper}")
print(f"  Ball color: {ball_color}")
print(f"  Video: {video_path}")
