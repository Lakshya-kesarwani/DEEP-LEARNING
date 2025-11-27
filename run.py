import cv2
import numpy as np
import mediapipe as mp
from math import atan2, degrees
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import time
from  config import MODEL_PATH, INPUT_H, INPUT_W, PADDING, SCREEN_W, SCREEN_H, SCREEN_W_MM, SCREEN_H_MM, DIST_MM, LEFT_EYE, RIGHT_EYE
from utils import crop_eye, normalized_to_pixel, compute_angle


def pre_processing(frame: np.ndarray,lm):
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
        print("Pre-processing complete. Ready to start the webcam.")
        return inp
    print("Pre-processing complete. Ready to start the webcam.")
    return None

def get_results(model,inp):
    pred = model.predict(inp, verbose=0)[0]
    xn, yn = round(float(pred[0]), 6), round(float(pred[1]), 6)
    x_px, y_px = normalized_to_pixel(xn, yn)
    yaw, pitch = compute_angle(x_px, y_px)
    return xn, yn, x_px, y_px, yaw, pitch

def show_onscreen_landmarks(show_landmarks,lm,frame):
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
    cv2.circle(frame, (capped_x, capped_y), 15, (0, 0, 255), -1)

def text_onscreen(frame,yaw,pitch,show_landmarks):
    cv2.putText(frame, 
                f"Yaw (Left/Right): {yaw:.1f} deg",
                (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,255,0), 2)
    cv2.putText(frame, 
                f"Pitch (Up/Down): {pitch:.1f} deg",
                (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,255,0), 2)
    cv2.putText(frame,
                f"Press 'l' to toggle landmarks: {'ON' if show_landmarks else 'OFF'}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2)


print("-----------------------Loading model...------------------------")
model = load_model(MODEL_PATH, compile=False)
print("-----------------------Model loaded.----------------------")

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")
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
        print("-------------------Face landmarks detection.------------------------")
        
        inp = pre_processing(frame, lm)
        if inp is None: 
            print("Eye pre-processing failed. Skipping frame.")
            continue
        
        print("-------------------Predicting gaze direction...------------------------")
        xn, yn, x_px, y_px, yaw, pitch = get_results(model, inp)
        pointer_coords = (int(x_px), int(y_px))
        show_onscreen_landmarks(show_landmarks, lm, frame)
        draw_gaze_pointer(frame, pointer_coords)
    
    print("-------------------Frame processed. Displaying results...--------------------")
    frame = cv2.flip(frame, 1)
    text_onscreen(frame, yaw, pitch, show_landmarks)
    cv2.imshow("Real-Time Gaze Angle", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        show_landmarks = not show_landmarks

cap.release()
cv2.destroyAllWindows()