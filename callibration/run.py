import cv2
import numpy as np
import mediapipe as mp
from math import atan2, degrees
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import time
from config import MODEL_PATH, INPUT_H, INPUT_W, PADDING, SCREEN_W, SCREEN_H, SCREEN_W_MM, SCREEN_H_MM, DIST_MM, LEFT_EYE, RIGHT_EYE, CALIBRATION_TIME
from utils import crop_eye, normalized_to_pixel, compute_angle, load_calibration, save_calibration, apply_calibration

SMOOTHING_FACTOR = 0.3 
_prev_x_px = None
_prev_y_px = None

# Calibration corners
CALIBRATION_POINTS = {
    "top_left": (50, 50),
    "top_right": (SCREEN_W - 50, 50),
    "bottom_left": (50, SCREEN_H - 50),
    "bottom_right": (SCREEN_W - 50, SCREEN_H - 50),
    "center": (SCREEN_W // 2, SCREEN_H // 2)
}

def pre_processing(frame: np.ndarray, lm):
    left_eye = crop_eye(frame, lm, LEFT_EYE)
    right_eye = crop_eye(frame, lm, RIGHT_EYE)

    if left_eye is not None and right_eye is not None:
        h = min(left_eye.shape[0], right_eye.shape[0])
        left_eye = cv2.resize(left_eye, (left_eye.shape[1], h))
        right_eye = cv2.resize(right_eye, (right_eye.shape[1], h))

        combined = cv2.hconcat([left_eye, right_eye])
        inp = cv2.resize(combined, (INPUT_W, INPUT_H))
        inp = inp.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)
        return inp
    return None

def get_results(model, inp):
    pred = model.predict(inp, verbose=0)[0]
    xn, yn = round(float(pred[0]), 6), round(float(pred[1]), 6)
    x_px, y_px = normalized_to_pixel(xn, yn)
    yaw, pitch = compute_angle(x_px, y_px)
    return xn, yn, x_px, y_px, yaw, pitch

def show_onscreen_landmarks(show_landmarks, lm, frame):
    if show_landmarks:
        for landmark in lm:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

def draw_gaze_pointer(frame, pointer_coords):
    if pointer_coords is None:
        return
    x_px, y_px = pointer_coords
    capped_x = max(0, min(int(x_px), SCREEN_W - 1))
    capped_y = max(0, min(int(y_px), SCREEN_H - 1))
    cv2.circle(frame, (capped_x-1089, capped_y), 15, (0, 0, 255), -1)

def text_onscreen(frame, yaw, pitch, show_landmarks):
    cv2.putText(frame, 
                f"Yaw (Left/Right): {yaw:.1f} deg",
                (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)
    cv2.putText(frame, 
                f"Pitch (Up/Down): {pitch:.1f} deg",
                (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)
    cv2.putText(frame,
                f"Press 'l' to toggle landmarks: {'ON' if show_landmarks else 'OFF'}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"(0, 0)", (SCREEN_W - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"(0, {SCREEN_H})", (30, SCREEN_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"({SCREEN_W}, {SCREEN_H})", (SCREEN_W - 200, SCREEN_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"({SCREEN_W}, 0)", (SCREEN_W - 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

def calibration_mode(model, mp_face, cap):
    """Calibration mode: collect data at 5 points"""
    calibration_data = {}
    
    for point_name, (target_x, target_y) in CALIBRATION_POINTS.items():
        print(f"\n{'='*60}")
        print(f"Calibrating {point_name.upper()}: ({target_x}, {target_y})")
        print(f"Look at the red circle for {CALIBRATION_TIME} seconds...")
        print(f"{'='*60}")
        
        collected_points = []
        start_time = time.time()
        
        while time.time() - start_time < CALIBRATION_TIME:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_face.process(rgb)
            
            # Draw calibration target
            frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))
            cv2.circle(frame, (target_x, target_y), 30, (0, 0, 255), -1)
            
            elapsed = time.time() - start_time
            remaining = max(0, CALIBRATION_TIME - elapsed)
            cv2.putText(frame, f"Time: {remaining:.1f}s", (SCREEN_W // 2 - 100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                inp = pre_processing(frame, lm)
                if inp is not None:
                    xn, yn, x_px, y_px, _, _ = get_results(model, inp)
                    collected_points.append((x_px, y_px))
                    cv2.circle(frame, (int(x_px), int(y_px)), 15, (255, 0, 0), 2)
            
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
        
        # Calculate average gaze point during calibration
        if collected_points:
            avg_x = np.mean([p[0] for p in collected_points])
            avg_y = np.mean([p[1] for p in collected_points])
            
            # Calculate correction offset
            offset_x = target_x - avg_x
            offset_y = target_y - avg_y
            
            calibration_data[point_name] = (offset_x, offset_y)
            print(f"Calibrated {point_name}: offset=({offset_x:.2f}, {offset_y:.2f})")
    
    cv2.destroyWindow("Calibration")
    return calibration_data

# ...existing code...
print("-----------------------Loading model...------------------------")
model = load_model(MODEL_PATH, compile=False)
print("-----------------------Model loaded.----------------------")

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load existing calibration or start fresh
load_calibration()

cap = cv2.VideoCapture(0)
print("Webcam started.")
print("Press 'c' to start calibration, 'q' to quit.")
show_landmarks = True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    print("-------------------Processing frame...--------------------")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)
    yaw, pitch = 0, 0
    pointer_coords = None

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark        
        inp = pre_processing(frame, lm)
        if inp is None: 
            print("Eye pre-processing failed. Skipping frame.")
            continue
        
        xn, yn, x_px, y_px, yaw, pitch = get_results(model, inp)
        pointer_coords = (int(x_px), int(y_px))
        show_onscreen_landmarks(show_landmarks, lm, frame)
        draw_gaze_pointer(frame, pointer_coords)
    frame = frame.flip(1)
    frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))
    text_onscreen(frame, yaw, pitch, show_landmarks)
    cv2.putText(frame, "Press 'c' to calibrate", (30, SCREEN_H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Real-Time Gaze Angle", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        show_landmarks = not show_landmarks
    elif key == ord('c'):
        print("\nStarting calibration...")
        calib_data = calibration_mode(model, mp_face, cap)
        if calib_data:
            save_calibration(calib_data)
            load_calibration()

cap.release()
cv2.destroyAllWindows()