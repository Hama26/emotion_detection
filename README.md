# Real-Time Emotion Detection Application (TFLite & MediaPipe)

This application uses a webcam to perform real-time emotion detection using a pre-trained TFLite model (`emotion.tflite`) and MediaPipe for face detection. Detected emotions are displayed on the video feed and streamed over a socket connection.

## Prerequisites

*   Python 3.8+
*   pip (Python package installer)
*   A webcam

## Setup

1.  **Clone the repository or download the files** into a local directory.

2.  **Place your TFLite model file**:
    *   Ensure your trained TFLite emotion detection model, `emotion.tflite`, is placed in the root of the project directory.

3.  **Install dependencies**:
    Open a terminal or command prompt in the project directory and run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `tensorflow`, `opencv-python`, `numpy`, `mediapipe`, `matplotlib`, and `pandas`.

## Running the Application

1.  Navigate to the project directory in your terminal.
2.  Run the main application script:
    ```bash
    python emotion_app.py
    ```
3.  A window will open showing the webcam feed with detected emotions. Emotion data will also be streamed via a socket (default: localhost, port 9999).
4.  Press 'q' to quit the application.

## Socket Streaming

*   The application starts a socket server on `localhost` (127.0.0.1) at port `9999` by default.
*   It streams the primary detected emotion as a string (e.g., "happiness", "sadness") whenever a new emotion is detected.
*   A client can connect to this socket to receive the emotion data.
