import cv2
import os
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]

def extract_eye_region(image, landmarks, eye_landmarks, padding=30):
    h, w, _ = image.shape
    eye_points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_landmarks]
    x_coords, y_coords = zip(*eye_points)
    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, h)
    return image[y_min:y_max, x_min:x_max] if (x_max - x_min > 0 and y_max - y_min > 0) else None

def process_and_save_eyes(input_dir, output_dir):
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue

        save_label_path = os.path.join(output_dir, label)
        os.makedirs(save_label_path, exist_ok=True)

        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                print(f"No face detected in: {img_path}")
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = extract_eye_region(image, landmarks, LEFT_EYE_LANDMARKS)
            right_eye = extract_eye_region(image, landmarks, RIGHT_EYE_LANDMARKS)

            if left_eye is not None and right_eye is not None:
                target_height = min(left_eye.shape[0], right_eye.shape[0])
                left_eye = cv2.resize(left_eye, (left_eye.shape[1], target_height))
                right_eye = cv2.resize(right_eye, (right_eye.shape[1], target_height))

                if len(left_eye.shape) == 2:
                    left_eye = cv2.cvtColor(left_eye, cv2.COLOR_GRAY2BGR)
                if len(right_eye.shape) == 2:
                    right_eye = cv2.cvtColor(right_eye, cv2.COLOR_GRAY2BGR)

                combined = cv2.hconcat([left_eye, right_eye])
                save_path = os.path.join(save_label_path, filename)
                # decrease quality to reduce size
                

                cv2.imwrite(save_path, combined)
            else:
                print(f"Eye extraction failed for {img_path}")
        
if __name__ == '__main__':
    input_dataset = 'dataset'
    output_dataset = 'dataset_cropped'
    process_and_save_eyes(input_dataset, output_dataset)