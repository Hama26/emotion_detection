import cv2
import time
from emotion_detector import predict_emotion_tflite, load_emotion_model_tflite, EMOTION_LABELS
from socket_streamer import EmotionSocketStreamer
import mediapipe as mp

# Configuration
CASCADE_PATH = 'haarcascade_frontalface_default.xml' # Make sure this file is in the root directory
WEBCAM_ID = 0 # Default webcam. Change if you have multiple cameras.
SOCKET_HOST = '127.0.0.1'
SOCKET_PORT = 9999

# Frame processing parameters
FRAME_SKIP = 2 # Process every Nth frame to improve performance, 1 means process every frame

def main():
    # --- Initialization ---
    mp_face_detection = mp.solutions.face_detection

    print("Starting Real-Time Emotion Detection Application...")

    # 1. Load Face Cascade Classifier
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade classifier from {CASCADE_PATH}. Please ensure the file exists and is accessible. Exiting.")
        return

    # Load TFLite emotion detection model
    interpreter = load_emotion_model_tflite() # Use the directly imported function
    if interpreter is None:
        print("Error: TFLite emotion detection model could not be loaded. Exiting.")
        return
    print("TFLite emotion detection model loaded successfully.")

    # Initialize socket streamer
    streamer = EmotionSocketStreamer(SOCKET_HOST, SOCKET_PORT)
    streamer.start_server()
    print(f"Socket server started on {SOCKET_HOST}:{SOCKET_PORT}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        streamer.stop_server()
        return
    print("Webcam initialized successfully.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {frame_width}x{frame_height}")

    last_emotion_sent_time = time.time()
    emotion_send_interval = 1
    last_known_emotion = None

    try:
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
            
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam. Exiting loop.")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                frame_rgb.flags.writeable = False
                results = face_detection.process(frame_rgb)
                frame_rgb.flags.writeable = True

                detected_emotions_in_frame = []

                if results.detections:
                    for detection in results.detections:
                        bbox_relative = detection.location_data.relative_bounding_box
                        
                        xmin = int(bbox_relative.xmin * frame_width)
                        ymin = int(bbox_relative.ymin * frame_height)
                        width = int(bbox_relative.width * frame_width)
                        height = int(bbox_relative.height * frame_height)

                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(frame_width, xmin + width)
                        ymax = min(frame_height, ymin + height)
                        
                        if xmax > xmin and ymax > ymin:
                            face_roi_bgr = frame_bgr[ymin:ymax, xmin:xmax]
                            
                            if face_roi_bgr.size > 0:
                                emotion = predict_emotion_tflite(face_roi_bgr) # Call imported function directly
                                last_known_emotion = emotion # Store the latest detected emotion
                                detected_emotions_in_frame.append(emotion) # Keep for display purposes

                                cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.putText(frame_bgr, str(emotion), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                pass # face_roi_bgr has no size
                        else:
                            pass # Bounding box has no area
                
                # Send last known emotion every second
                current_time = time.time()
                if last_known_emotion is not None and (current_time - last_emotion_sent_time >= emotion_send_interval):
                    print(f"[EmotionApp] Sending via socket: {last_known_emotion}")
                    streamer.send_emotion(str(last_known_emotion))
                    last_emotion_sent_time = current_time

                cv2.imshow('Real-time Emotion Detection (MediaPipe & TFLite)', frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit signal received. Shutting down.")
                    break
    finally:
        print("Releasing resources...")
        cap.release()
        print("Webcam released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        if streamer.running:
            streamer.stop_server()
            print("Socket streamer stopped.")
        print("Application finished.")

if __name__ == '__main__':
    main()
