import cv2
import numpy as np
import csv

# ===============================
# Load table calibration
# ===============================
table_corners = np.load("table_corners.npy").astype(np.float32)
print("Loaded table corners:", table_corners)
TABLE_WIDTH  = 1.525   # meters
TABLE_LENGTH = 2.74    # meters

table_pts = np.array([
    [0.0, 0.0],                 # bottom-left
    [TABLE_WIDTH, 0.0],          # bottom-right
    [TABLE_WIDTH, TABLE_LENGTH], # top-right
    [0.0, TABLE_LENGTH]          # top-left
], dtype=np.float32)

H, _ = cv2.findHomography(table_corners, table_pts)

# ===============================
# Video
# ===============================
cap = cv2.VideoCapture("low_input.mp4")
frame_id = 0

# Resize factor for display
DISPLAY_SCALE = 0.6

# ===============================
# CSV output
# ===============================
csv_file = open("ball_coordinates.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "x_table", "y_table"])

# ===============================
# Main loop
# ===============================

prev_y = None
prev_vy = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ===============================
    # Ball mask (yellow-orange)
    # ===============================
    lower = np.array([15, 150, 150])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # ===============================
    # Remove green table
    # ===============================
    green_lower = np.array([35, 80, 80])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))

    # ===============================
    # Clean mask
    # ===============================
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ===============================
    # Contours
    # ===============================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy = None, None

    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 500:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

            if 0.7 < aspect_ratio < 1.3:
                cx = x + w // 2
                cy = y + h // 2
                break

    # ===============================
    # Convert to table coordinates
    # ===============================
    if cx is not None:
        # compute vertical velocity
        bounce_detected = False

        if prev_y is not None:
            vy = cy - prev_y  # +ve = moving down, -ve = moving up

            if prev_vy is not None:
                # DOWN → UP transition = table touch
                if prev_vy > 2 and vy < -2:
                    bounce_detected = True

            prev_vy = vy

        prev_y = cy

        # ===============================
        # If bounce detected → output coords
        # ===============================
        if bounce_detected:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            table_pt = cv2.perspectiveTransform(pt, H)
            xt, yt = table_pt[0][0]

            writer.writerow([frame_id, xt, yt])
            print(f"BOUNCE at frame {frame_id}: x={xt:.3f}, y={yt:.3f}")

            # Visual confirmation
            cv2.circle(frame, (cx, cy), 12, (255, 0, 0), 3)

        # Always draw tracking dot
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


    # ===============================
    # Display
    # ===============================
    frame_disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    mask_disp = cv2.resize(mask, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Ball Tracking (Table Coordinates)", frame_disp)
    cv2.imshow("Mask (Debug)", mask_disp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    if cv2.waitKey(delay) & 0xFF == 27:
        break

    frame_id += 1


# ===============================
# Cleanup
# ===============================
cap.release()
csv_file.close()
cv2.destroyAllWindows()
