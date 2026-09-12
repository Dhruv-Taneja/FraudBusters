import cv2
import numpy as np

# In preprocessing.py: Add this function

def check_image_quality(image, blur_threshold=100.0):
    """
    Evaluates image quality (blur and exposure).
    Returns:
        is_valid (bool): Whether the image is good enough for processing.
        reason (str): Reason for failure if any.
        blur_score (float): Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Measure edge variance (blur detection)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Measure mean brightness
    mean_brightness = np.mean(gray)
    
    if blur_score < blur_threshold:
        return False, f"Image is too blurry (Sharpness Score: {blur_score:.2f})", blur_score
        
    if mean_brightness < 35:
        return False, f"Image is underexposed/too dark (Brightness: {mean_brightness:.1f})", blur_score
        
    if mean_brightness > 235:
        return False, f"Image is overexposed/glared (Brightness: {mean_brightness:.1f})", blur_score

    return True, "Quality passed", blur_score

def deskew_and_crop(image):
    """
    Deskew and crop the document using OpenCV.
    Args:
        image: A numpy array representing the image (BGR).
    Returns:
        The deskewed and cropped image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour which is presumably the document
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If the approximated contour has 4 points, we can apply perspective transform
    if len(approx) == 4:
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        
        # Compute the width and height of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Construct the set of destination points to obtain a "birds eye view"
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    else:
        # Fallback if we don't find a 4-point contour
        return image

def correct_illumination(image):
    """
    Apply adaptive histogram equalization (CLAHE) to fix shadows/glares.
    Args:
        image: A numpy array representing the image (BGR).
    Returns:
        Illumination corrected image.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

def preprocess_image(image_path):
    """
    Runs the full preprocessing pipeline.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    deskewed = deskew_and_crop(image)
    corrected = correct_illumination(deskewed)
    return corrected
