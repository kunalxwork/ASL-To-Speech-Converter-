# Real-Time American Sign Language (ASL) to Speech

This project is a real-time ASL-to-speech converter built by (Your Name 1), (Your Name 2), and (Your Name 3).

It uses a modern, lightweight pipeline to detect hand gestures from a webcam, classify them using a custom-trained neural network, and convert the resulting text to speech, all in real-time.

## 🚀 Features

* **High-Speed Detection:** Uses **Google's MediaPipe** to extract 21 hand keypoints, which is much faster than traditional CNNs.
* **Robust Classification:** Employs a **TensorFlow Lite (TFLite)** quantized model for highly efficient, on-device inference.
* **Real-Time Audio:** Features a "hold-to-confirm" logic that builds a sentence and uses a multi-threaded `pygame` engine to play audio without freezing the webcam feed.

## 🔧 How It Works

This project is split into three main parts:

1.  **`collect_hand_data.py`**: A script to capture hand landmarks from a webcam. It saves the 42 normalized keypoints (21 x/y) for each sign into a `keypoint.csv` file.
2.  **`keypoint_classification.ipynb`**: A Jupyter Notebook that loads the `keypoint.csv`, trains a TensorFlow/Keras neural network on the data, and evaluates its accuracy.
3.  **`real_time_detector.py`**: The final application. It uses MediaPipe to get keypoints, feeds them into the trained TFLite model for prediction, and speaks the confirmed letter.

## 🛠️ How to Run This Project

### 1. Installation

Clone the repository and install the required libraries:

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

### 2. (Optional) Re-train the Model

You can use the included `.tflite` model, or train your own:

1.  Run the data collection script. Press a key to record samples for each letter (A-Z).
    ```bash
    python collect_hand_data.py
    ```
2.  Run the `keypoint_classification.ipynb` notebook to create a new `.keras` model.
3.  Run the conversion script to create a new `.tflite` model (this is optional, as the main script can also use the `.keras` file).
    ```bash
    python convert_v2.py
    ```

### 3. Run the Detector

This is the main file. It will load the audio cache and the TFLite model.

```bash
python real_time_detector.py
```

Show a hand sign to the camera. Hold it for ~1.5 seconds to confirm the letter and add it to the sentence.