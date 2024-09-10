import numpy as np, opennsfw2, cv2, modules.globals
from PIL import Image
from modules.typing import Frame

# Threshold for NSFW content probability
MAX_PROBABILITY = 0.85

# Preload the NSFW model for efficiency
model = None

def predict_frame(target_frame: Frame) -> bool:
    """
    Predicts whether a given frame contains NSFW content.

    Args:
        target_frame (Frame): The image frame to be analyzed.

    Returns:
        bool: True if NSFW content probability exceeds the threshold, False otherwise.
    """
    global model
    # Apply color correction if enabled
    if modules.globals.color_correction:
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to PIL image and preprocess
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    
    # Load model if not already loaded
    if model is None:
        model = opennsfw2.make_open_nsfw_model()

    # Predict and return NSFW likelihood
    views = np.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY

def predict_image(target_path: str) -> bool:
    """
    Predicts whether an image at the specified path contains NSFW content.

    Args:
        target_path (str): Path to the image file.

    Returns:
        bool: True if NSFW content probability exceeds the threshold, False otherwise.
    """
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY

def predict_video(target_path: str) -> bool:
    """
    Predicts whether a video at the specified path contains NSFW content based on sampled frames.

    Args:
        target_path (str): Path to the video file.

    Returns:
        bool: True if any frame exceeds the NSFW probability threshold, False otherwise.
    """
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(prob > MAX_PROBABILITY for prob in probabilities)
