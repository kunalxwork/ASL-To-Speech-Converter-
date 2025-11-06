import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import time
import os
import threading
from gtts import gTTS
import pygame  # <--- NEW IMPORT

# --- Configuration ---
TFLITE_MODEL_PATH = 'model/keypoint_classifier/keypoint_classifier.tflite'
NUM_CLASSES = 26
CONFIDENCE_THRESHOLD = 0.10
FRAMES_TO_SAMPLE = 15

# --- Audio Caching ---
AUDIO_CACHE_DIR = "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
print("Checking audio cache...")
for i in range(NUM_CLASSES):
    letter = chr(ord('A') + i)
    audio_file = os.path.join(AUDIO_CACHE_DIR, f"{letter}.mp3")
    if not os.path.exists(audio_file):
        print(f"Creating audio for: {letter}")
        tts = gTTS(text=letter, lang='en')
        tts.save(audio_file)
print("Audio cache complete.")

# --- NEW Speak Function (using pygame) ---
def speak(letter):
    try:
        audio_file = os.path.abspath(os.path.join(AUDIO_CACHE_DIR, f"{letter}.mp3"))
        if os.path.exists(audio_file):
            # pygame.mixer.Sound() loads the file and plays it
            pygame.mixer.Sound(audio_file).play()
        else:
            print(f"Error: Missing audio file for {letter}")
    except Exception as e:
        print(f"Error playing sound: {e}")

# --- Load the TFLite Model ---
print(f"Loading TFLite model from {TFLITE_MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded successfully!")

IS_QUANTIZED = input_details[0]['dtype'] == np.int8
if IS_QUANTIZED:
    print("Model is 8-bit quantized.")

# --- Create Label Map ---
label_map = [chr(ord('A') + i) for i in range(NUM_CLASSES)]

# --- MediaPipe Hand Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Normalization Function ---
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
        image, landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
    )

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    
    # --- NEW: Initialize pygame mixer ---
    pygame.mixer.init()
    # ------------------------------------
    
    sentence = ""
    current_letter = ""
    last_spoken_letter = ""
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        prediction_text = "No hand detected"
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            draw_hand(frame, hand_landmarks)
            keypoints = normalize_landmarks(hand_landmarks, frame_width, frame_height)
            
            if keypoints is not None:
                model_input = np.array([keypoints], dtype=np.float32)

                if IS_QUANTIZED:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    model_input = (model_input / input_scale) + input_zero_point
                    model_input = model_input.astype(np.int8)

                interpreter.set_tensor(input_details[0]['index'], model_input)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                
                if IS_QUANTIZED:
                    output_scale, output_zero_point = output_details[0]['quantization']
                    prediction = (prediction.astype(np.float32) - output_zero_point) * output_scale
                
                predicted_class_id = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_letter = label_map[predicted_class_id]
                    prediction_text = f"Prediction: {predicted_letter} (Conf: {confidence:.2f})"
                    
                    if predicted_letter != current_letter:
                        current_letter = predicted_letter
                        frame_count = 0
                    else:
                        frame_count += 1
                        
                    if frame_count == FRAMES_TO_SAMPLE:
                        if current_letter != last_spoken_letter:
                            sentence += current_letter
                            last_spoken_letter = current_letter
                            # Still use a thread to be safe
                            threading.Thread(target=speak, args=(current_letter,), daemon=True).start()
                            
                else: 
                    prediction_text = "Low Confidence"
                    current_letter = ""
                    frame_count = 0
                    
            else:
                prediction_text = "Hand detected, but normalization failed"
                    
        else: 
            current_letter = ""
            frame_count = 0
            last_spoken_letter = "" 
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, prediction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (frame_width - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (0, frame_height - 50), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.putText(frame, f"Sentence: {sentence}", (10, frame_height - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('ASL Keypoint Detection (TFLite)', frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit() # <-- NEW: Quit pygame
    print(f"Final Sentence: {sentence}")

if __name__ == "__main__":
    main()
