# import cv2
# import numpy as np

# VIDEO_PATH = "input.mp4"
# cap = cv2.VideoCapture(VIDEO_PATH)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # ---------------------------
#     # Convert to HSV
#     # ---------------------------
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # ---------------------------
#     # TABLE COLOR RANGE (EDIT IF NEEDED)
#     # Typical blue/green table
#     # ---------------------------
#     lower_table = np.array([80, 40, 40])
#     upper_table = np.array([140, 255, 255])

#     table_mask = cv2.inRange(hsv, lower_table, upper_table)

#     # ---------------------------
#     # BALL COLOR RANGE (orange/yellow)
#     # ---------------------------
#     lower_ball = np.array([5, 150, 150])
#     upper_ball = np.array([35, 255, 255])

#     ball_mask = cv2.inRange(hsv, lower_ball, upper_ball)

#     # ---------------------------
#     # COMBINE MASKS
#     # ---------------------------
#     mask = cv2.bitwise_or(table_mask, ball_mask)

#     # Remove noise
#     kernel = np.ones((5,5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     # ---------------------------
#     # APPLY MASK
#     # ---------------------------
#     result = cv2.bitwise_and(frame, frame, mask=mask)

#     # Background automatically becomes black
#     cv2.imshow("Table + Ball Only", result)
#     cv2.imshow("Mask", mask)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
