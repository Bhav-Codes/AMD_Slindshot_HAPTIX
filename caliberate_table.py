"""Run this to feed table coordinates"""
import cv2
import numpy as np

video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video")
    exit()

points = []

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow("Select Table Corners", frame)

cv2.imshow("Select Table Corners", frame)
cv2.setMouseCallback("Select Table Corners", mouse_click)

print("Click table corners in this order:")
print("1. Bottom-Left")
print("2. Bottom-Right")
print("3. Top-Right")
print("4. Top-Left")
print("Press ESC after clicking 4 points")

while True:
    cv2.imshow("Select Table Corners", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

if len(points) != 4:
    print("ERROR: You must click exactly 4 points")
    exit()

np.save("table_corners.npy", np.array(points, dtype=np.float32))
print("Saved table corners to table_corners.npy")
