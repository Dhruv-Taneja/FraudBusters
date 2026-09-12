import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def perform_ela(image_path, quality=90):
    """
    Error Level Analysis (ELA).
    Highlights discrepancies in compression levels, indicative of image manipulation.
    Returns a score (0.0 to 1.0) representing the likelihood of alteration based on ELA anomalies.
    """
    original = Image.open(image_path).convert('RGB')
    
    # Save the image at a known quality level
    temp_filename = 'temp_ela.jpg'
    original.save(temp_filename, 'JPEG', quality=quality)
    
    # Open the resaved image
    resaved = Image.open(temp_filename)
    
    # Calculate the difference between the original and resaved image
    ela_image = ImageChops.difference(original, resaved)
    
    # Get the extrema (min and max values) to scale the image
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
        
    # Enhance the ELA image
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    os.remove(temp_filename)
    
    # Calculate a simple "alteration score" based on the variance of the ELA image
    # Forged regions will have higher error values
    ela_array = np.array(ela_image)
    gray_ela = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
    
    # A high variance in ELA often points to splicing/forgery
    variance = np.var(gray_ela)
    
    # Normalize variance to a 0.0 - 1.0 score (this threshold needs tuning based on real data)
    score = min(variance / 1000.0, 1.0)
    
    return score, ela_image

def detect_forgery_cnn(image_path):
    """
    Placeholder for Pixel-Level Forgery Detection (e.g., Mantra-Net).
    In a real implementation, this would load a pre-trained CNN to detect copy-move and splicing.
    """
    # Mock output
    # In reality, this would return a probability mask or an aggregate score
    print("Mantra-Net (CNN Forgery Detection) running... (Mock)")
    return 0.15  # Returning a mock low suspicion score

def check_font_inconsistency(patches):
    """
    Placeholder for Font Inconsistency using a Siamese Neural Network.
    Takes structural embeddings of font patches and compares them.
    """
    # Mock output
    print("Siamese Network checking font structural embeddings... (Mock)")
    if len(patches) < 2:
        return 0.0
    # In reality, extract features from patches and compute cosine distance
    return 0.05 # Mock score

def get_tampering_score(image_path):
    """
    Aggregate various tampering detection methods.
    """
    ela_score, _ = perform_ela(image_path)
    cnn_score = detect_forgery_cnn(image_path)
    
    # Combine scores (weighted average)
    final_score = (ela_score * 0.4) + (cnn_score * 0.6)
    return final_score
