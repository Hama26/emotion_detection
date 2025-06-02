import tensorflow as tf
import numpy as np
import cv2

# Define the expected input size for the TFLite model
IMG_SIZE = (48, 48)

# Define emotion labels based on the provided example code
# These correspond to the output of the TFLite model
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness", "anger", 
    "disgust", "fear", "contempt", "Unknown", "NF" # NF = Not Face
]

TFLITE_MODEL_PATH = './emotion_mobilenet_v2.tflite'

# Global variables for TFLite interpreter and details
_interpreter = None
_input_details = None
_output_details = None

def load_emotion_model_tflite(model_path: str = TFLITE_MODEL_PATH):
    """Loads the TFLite emotion detection model and allocates tensors."""
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        try:
            _interpreter = tf.lite.Interpreter(model_path=model_path)
            _interpreter.allocate_tensors()
            _input_details = _interpreter.get_input_details()
            _output_details = _interpreter.get_output_details()
            print(f"TFLite model loaded successfully from {model_path}")
            # Print model input and output details for verification
            # print("Input Details:", _input_details)
            # print("Output Details:", _output_details)
        except Exception as e:
            print(f"Error loading TFLite model from {model_path}: {e}")
            print("Please ensure 'emotion.tflite' is in the correct directory and is a valid TFLite model.")
            _interpreter = None
    return _interpreter

def preprocess_face_for_tflite(face_image_bgr):
    """
    Preprocesses the face image for the TFLite emotion detection model.
    Args:
        face_image_bgr: A BGR image (NumPy array) of the detected face.
    Returns:
        A preprocessed image (NumPy array) ready for the TFLite model, or None if preprocessing fails.
    """
    if face_image_bgr is None or face_image_bgr.size == 0:
        print("Error: Empty face image received for preprocessing.")
        return None
    try:
        # Resize to model's expected input size (48x48)
        img_resized = cv2.resize(face_image_bgr, IMG_SIZE)
        
        # Convert to Grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Repeat grayscale channel 3 times to make it 3-channel (as per example)
        img_3channel = np.repeat(img_gray[..., np.newaxis], 3, -1)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_3channel.astype(np.float32) / 255.0
        
        # Expand dimensions to create a batch of 1 (e.g., (1, 48, 48, 3))
        # Ensure this matches _input_details[0]['shape'] which is often [1, height, width, channels]
        preprocessed_image = np.expand_dims(img_normalized, axis=0)
        
        return preprocessed_image
    except Exception as e:
        print(f"Error preprocessing face for TFLite: {e}")
        return None

def predict_emotion_tflite(face_image_bgr):
    """
    Predicts the emotion from a BGR face image using the TFLite model.
    Args:
        face_image_bgr: A BGR image (NumPy array) of the detected face.
    Returns:
        A string representing the detected emotion, or "Error" if prediction fails.
    """
    interpreter = load_emotion_model_tflite() # Ensure model is loaded
    if interpreter is None or _input_details is None or _output_details is None:
        return "Model not loaded"

    preprocessed_face = preprocess_face_for_tflite(face_image_bgr)
    if preprocessed_face is None:
        return "Preprocessing failed"

    try:
        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(_input_details[0]['index'], preprocessed_face)
        # Run inference
        interpreter.invoke()
        # Extract output data from the tensor
        output_data = interpreter.get_tensor(_output_details[0]['index'])
        
        # Get the index of the highest probability
        pred_probabilities = np.squeeze(output_data) # Squeeze to remove batch dimension if present
        emotion_index = np.argmax(pred_probabilities)
        
        if 0 <= emotion_index < len(EMOTION_LABELS):
            detected_emotion = EMOTION_LABELS[emotion_index]
        else:
            print(f"Error: Predicted emotion index {emotion_index} is out of bounds for labels.")
            detected_emotion = "Prediction index error"
            
        return detected_emotion
    except Exception as e:
        print(f"Error during TFLite emotion prediction: {e}")
        return "Prediction error"

if __name__ == '__main__':
    print("Testing TFLite emotion_detector.py...")
    tflite_interpreter = load_emotion_model_tflite() 

    if tflite_interpreter:
        print("TFLite Model loaded. Creating a dummy image for prediction test...")
        # Create a dummy BGR image (e.g., black image)
        dummy_face_bgr = np.zeros((100, 100, 3), dtype=np.uint8) 
        
        print("Testing preprocessing...")
        preprocessed_dummy = preprocess_face_for_tflite(dummy_face_bgr)
        if preprocessed_dummy is not None:
            print(f"Preprocessed dummy image shape: {preprocessed_dummy.shape}, dtype: {preprocessed_dummy.dtype}")
            # Verify input shape compatibility
            expected_shape = tuple(_input_details[0]['shape'])
            if preprocessed_dummy.shape == expected_shape:
                print(f"Preprocessed shape matches model input shape {expected_shape}.")
                emotion = predict_emotion_tflite(dummy_face_bgr)
                print(f"Predicted emotion on dummy image: {emotion}")
            else:
                print(f"Shape mismatch: Preprocessed shape {preprocessed_dummy.shape} vs Model input shape {expected_shape}")
        else:
            print("Preprocessing of dummy image failed.")
    else:
        print("TFLite Model could not be loaded. Skipping prediction test.")
    print("TFLite emotion detector script test finished.")

