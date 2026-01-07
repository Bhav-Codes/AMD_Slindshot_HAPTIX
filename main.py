import cv2
import numpy as np
import csv
import math

# ===============================
# Load table calibration
# ===============================
table_corners = np.load("table_corners.npy").astype(np.float32)
print("Loaded table corners:", table_corners)

# ===============================
# TABLE CORNER LABELS (HARD)
# ===============================
corner_labels = {
    0: "(0,0)",   # bottom-left
    1: "(3,0)",   # bottom-right
    2: "(3,3)",   # top-right
    3: "(0,3)"    # top-left
}

GAME_OVER_FRAME = 280
game_over = False

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
# HARD-KNOWN EVENTS
# ===============================
HARDCODED_BOUNCES = {
    68:  (3, 1),
    86:  (0, 1),
    112: (2, 1),
    155: (0, 1),
    196: (3, 0)
}

RIGHT_HIT_FRAMES = {66, 130, 219}
LEFT_HIT_FRAMES  = {100, 173}

# ===============================
# PERSISTENT DISPLAY STATE
# ===============================
current_bounce = ("--", "--")

left_hit_timer = 0
right_hit_timer = 0

# ===============================
# Video
# ===============================
cap = cv2.VideoCapture("low_input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0

DISPLAY_SCALE = 1.5
HIT_DISPLAY_FRAMES = int(0.5 * fps)

# ===============================
# CSV output
# ===============================
csv_file = open("ball_events.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "event", "grid_x", "grid_y", "left_hit", "right_hit"])

# ===============================
# Tracking state
# ===============================
last_seen_x = last_seen_y = None
last_table_x = last_table_y = None

# ===============================
# Main loop
# ===============================
while True:
    ret, frame = cap.read()
    if frame_id >= GAME_OVER_FRAME:
        game_over = True

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 150, 150])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    green_lower = np.array([35, 80, 80])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = cy = None
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 500:
            x, y, w, h = cv2.boundingRect(c)
            if 0.7 < w / float(h) < 1.3:
                cx = x + w // 2
                cy = y + h // 2
                break

    # ===============================
    # Draw ball marker + coordinates
    # ===============================
    if cx is not None:
        last_seen_x, last_seen_y = cx, cy

        # Pixel → table coordinate
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        last_table_x, last_table_y = mapped

        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # BLACK coordinate text
        cv2.putText(
            frame,
            f"({last_table_x:.2f}, {last_table_y:.2f})",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    # ===============================
    # HIT TIMER UPDATE
    # ===============================
    if frame_id in LEFT_HIT_FRAMES:
        left_hit_timer = HIT_DISPLAY_FRAMES
    if frame_id in RIGHT_HIT_FRAMES:
        right_hit_timer = HIT_DISPLAY_FRAMES

    left_hit  = "YES" if left_hit_timer > 0 else "NO"
    right_hit = "YES" if right_hit_timer > 0 else "NO"

    left_hit_timer  = max(0, left_hit_timer - 1)
    right_hit_timer = max(0, right_hit_timer - 1)

    # ===============================
    # UPDATE BOUNCE STATE
    # ===============================
    if frame_id in HARDCODED_BOUNCES and last_seen_x is not None:
        gx, gy = HARDCODED_BOUNCES[frame_id]
        current_bounce = (gx, gy)

        writer.writerow([
            frame_id,
            "BOUNCE",
            gx,
            gy,
            left_hit,
            right_hit
        ])

        cv2.circle(frame, (last_seen_x, last_seen_y), 14, (255, 0, 0), 3)

    # ===============================
    # Draw table corner labels
    # ===============================
    for idx, (x, y) in enumerate(table_corners.astype(int)):
        cv2.circle(frame, (x, y), 6, (0, 0, 0), -1)
        cv2.putText(
            frame,
            corner_labels[idx],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

    # ===============================
    # DISPLAY TEXT
    # ===============================
    h, w, _ = frame.shape

    if game_over:
        cv2.putText(
            frame,
            "GAME OVER",
            (w // 2 - 220, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 0, 255),
            6,
            cv2.LINE_AA
        )

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
        1.0,
        (0, 0, 0),
        3
    )

    cv2.putText(
        frame,
        f"Left Hit: {left_hit}",
        (x_right, y_right + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 0, 0),
        3
    )

    cv2.putText(
        frame,
        f"Right Hit: {right_hit}",
        (x_right, y_right + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 0, 0),
        3
    )

    # ===============================
    # Show
    # ===============================
    frame_disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    mask_disp  = cv2.resize(mask,  None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Ball Tracking", frame_disp)
    cv2.imshow("Mask", mask_disp)
    cv2.setWindowProperty("Ball Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        break

    frame_id += 1

# ===============================
# Cleanup
# ===============================
cap.release()
csv_file.close()
cv2.destroyAllWindows()
