from config import SCREEN_W, SCREEN_H, SCREEN_W_MM, SCREEN_H_MM, DIST_MM, PADDING
from math import atan2, degrees
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
    x_px = (xn + 1) / 2 * SCREEN_W
    y_px = (yn + 1) / 2 * SCREEN_H
    return x_px, y_px


def compute_angle(x_px, y_px):
    cx = SCREEN_W / 2
    cy = SCREEN_H / 2

    dx_px = x_px - cx
    dy_px = y_px - cy

    dx_mm = dx_px * (SCREEN_W_MM / SCREEN_W)
    dy_mm = dy_px * (SCREEN_H_MM / SCREEN_H)

    yaw = degrees(atan2(dx_mm, DIST_MM))     # left-right
    pitch = degrees(atan2(dy_mm, DIST_MM))   # up-down

    # Clamp -90..90
    yaw = max(-90, min(90, yaw))
    pitch = max(-90, min(90, pitch))

    return yaw, pitch

