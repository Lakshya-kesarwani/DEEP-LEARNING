from config import SCREEN_W, SCREEN_H, SCREEN_W_MM, SCREEN_H_MM, DIST_MM, PADDING, CALIB_OFFSETS, CALIBRATION_FILE
from math import atan2, degrees
import json
import os

# Smoothing parameters
SMOOTHING_FACTOR = 0.3
_prev_x_px = None
_prev_y_px = None

# Load calibration offsets
def load_calibration():
    global CALIB_OFFSETS
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            CALIB_OFFSETS = json.load(f)
        print(f"Calibration loaded: {CALIB_OFFSETS}")
    return CALIB_OFFSETS

def save_calibration(offsets):
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(offsets, f, indent=2)
    print(f"Calibration saved to {CALIBRATION_FILE}")

def crop_eye(image, landmarks, eye_landmarks):
    h, w, _ = image.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_landmarks]
    xs, ys = zip(*points)

    x1 = max(min(xs) - PADDING, 0)
    x2 = min(max(xs) + PADDING, w)
    y1 = max(min(ys) - PADDING, 0)
    y2 = min(max(ys) + PADDING, h)

    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None

    return image[y1:y2, x1:x2]

def normalized_to_pixel(xn, yn):
    global _prev_x_px, _prev_y_px
    
    x_px = (-xn + 1) / 2 * SCREEN_W
    y_px = (yn + 1) / 2 * SCREEN_H
    
    return x_px, y_px

def apply_calibration(x_px, y_px, corner_key):
    """Apply calibration offset based on which corner was calibrated"""
    if corner_key in CALIB_OFFSETS:
        offset_x, offset_y = CALIB_OFFSETS[corner_key]
        x_px += offset_x
        y_px += offset_y
    return x_px, y_px

def compute_angle(x_px, y_px):
    cx = SCREEN_W / 2
    cy = SCREEN_H / 2

    dx_px = x_px - cx
    dy_px = y_px - cy

    dx_mm = dx_px * (SCREEN_W_MM / SCREEN_W)
    dy_mm = dy_px * (SCREEN_H_MM / SCREEN_H)

    yaw = degrees(atan2(dx_mm, DIST_MM))
    pitch = degrees(atan2(dy_mm, DIST_MM))

    yaw = max(-90, min(90, yaw))
    pitch = max(-90, min(90, pitch))

    return yaw, pitch