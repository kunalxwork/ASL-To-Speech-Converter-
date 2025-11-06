import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# --- Configuration ---
CSV_FILE = 'model/keypoint_classifier/keypoint.csv'
NUM_CLASSES = 26  # A-Z
STARTING_CLASS_ID = 0 # 0=A, 1=B, etc.

# --- MediaPipe Hand Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

print("MediaPipe Hands model loaded.")

def get_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None

def normalize_landmarks(landmarks, image_width, image_height):
    landmark_array = np.empty((21, 2))
    for i, landmark in enumerate(landmarks.landmark):
        landmark_array[i] = [landmark.x * image_width, landmark.y * image_height]

    relative_landmarks = landmark_array - landmark_array[0]
    flattened_landmarks = relative_landmarks.flatten()
    
    max_val = np.max(np.abs(flattened_landmarks))
    if max_val == 0:
        return None
        
    normalized_landmarks = flattened_landmarks / max_val
    return normalized_landmarks

def draw_hand(image, landmarks):
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
    )

def main():
    cap = cv2.VideoCapture(0)
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    
    current_class_id = STARTING_CLASS_ID
    current_class_char = chr(ord('A') + current_class_id)
    
    print(f"--- Starting Data Collection ---")
    print(f"Press 'n' to move to the next letter.")
    print(f"Press 'q' to quit.")
    print(f"Press any other key (e.g., spacebar) to record.")
    print(f"----------------------------------")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        info_text = f"Recording for: '{current_class_char}' (ID: {current_class_id})"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        landmarks = get_landmarks(frame)
        keypoints = None # Initialize keypoints as None
        
        if landmarks:
            draw_hand(frame, landmarks)
            keypoints = normalize_landmarks(landmarks, frame_width, frame_height)
        
        # --- KEYPRESS LOGIC (MOVED OUTSIDE) ---
        # This now runs every frame, so the window won't freeze
        key = cv2.waitKey(10) & 0xFF 
        
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_class_id += 1
            if current_class_id >= NUM_CLASSES:
                print("All classes done!")
                break
            current_class_char = chr(ord('A') + current_class_id)
            print(f"Moved to next class: '{current_class_char}' (ID: {current_class_id})")
        
        # Check for "record" key AND if a hand was actually detected
        elif key != 255 and key != ord('n') and keypoints is not None:
            data_row = [current_class_id] + list(keypoints)
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
            print(f"Recorded sample for '{current_class_char}'")
        
        # Show info for "no hand"
        if not landmarks:
            cv2.putText(frame, "Show hand, then press 'space' to record.", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Hand Landmark Data Collection', frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection finished. CSV saved to: {CSV_FILE}")

if __name__ == "__main__":
    main()
