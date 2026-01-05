# import cv2
# import cvzone
# from cvzone.ColorModule import ColorFinder
# from cvzone.Utils import displayDetectionDatasetSamples
# import os
# import time

# # HSV values for ball color (tune if needed)
# hsvVals = {
#     'hmin': 4,
#     'smin': 116,
#     'vmin': 0,
#     'hmax': 21,
#     'smax': 255,
#     'vmax': 255
# }

# myColorFinder = ColorFinder(trackBar=False)

# # =========================
# # FUNCTIONS
# # =========================
# def read_frame(cap):
#     success, img = cap.read()
#     if not success:
#         return None
#     return img


# def find_color(img, hsvVals):
#     return myColorFinder.update(img, hsvVals)


# def find_contours(img, mask):
#     return cvzone.findContours(
#         img,
#         mask,
#         minArea=200,
#         maxArea=1000,
#         sort=True
#     )


# def generate_yolo_format(contours, img_shape):
#     """
#     YOLO format:
#     class_id x_center y_center width height
#     (all normalized)
#     """
#     annotations = []
#     h, w, _ = img_shape

#     for contour in contours:
#         x, y, bw, bh = contour['bbox']

#         x_center = (x + bw / 2) / w
#         y_center = (y + bh / 2) / h
#         width = bw / w
#         height = bh / h

#         annotations.append(f"0 {x_center} {y_center} {width} {height}")

#     return annotations


# def save_detections(img, annotations, output_folder):
#     timestamp = int(time.time() * 1000)

#     image_path = os.path.join(output_folder, f"{timestamp}.jpg")
#     label_path = os.path.join(output_folder, f"{timestamp}.txt")

#     cv2.imwrite(image_path, img)

#     with open(label_path, 'w') as file:
#         for ann in annotations:
#             file.write(ann + "\n")

import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import os
import time

# ======================
# INITIALIZATION
# ======================
hsvVals = {
    'hmin': 4, 'smin': 116, 'vmin': 0,
    'hmax': 21, 'smax': 255, 'vmax': 255
}

myColorFinder = ColorFinder(trackBar=False)

# ======================
# FUNCTIONS
# ======================
def read_frame(cap):
    success, img = cap.read()
    if not success:
        return None
    return img

def find_color(img):
    return myColorFinder.update(img, hsvVals)

def find_contours(img, mask):
    return cvzone.findContours(img, mask, minArea=200, maxArea=1000, sort=True)

import cv2

def generate_yolo_format(contours, img_shape):
    annotations = []
    img_h, img_w = img_shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h

        annotations.append(f"0 {x_center} {y_center} {width} {height}")

    return annotations

def save_detections(img, annotations, output_folder):
    timestamp = int(time.time() * 1000)
    image_path = os.path.join(output_folder, f"{timestamp}.jpg")
    label_path = os.path.join(output_folder, f"{timestamp}.txt")

    cv2.imwrite(image_path, img)
    with open(label_path, "w") as f:
        for ann in annotations:
            f.write(ann + "\n")
