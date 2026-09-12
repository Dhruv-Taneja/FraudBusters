try:
    from deepface import DeepFace
except ImportError:
    print("deepface is not installed. Face verification will be mocked.")

def extract_and_verify_faces(img1_path, img2_path):
    """
    Extract faces and verify identity using DeepFace (which supports ArcFace/MTCNN).
    Returns a similarity score (Cosine).
    """
    try:
        # DeepFace handles face extraction (default is OpenCV or MTCNN based on backend)
        # and feature embedding (we specify ArcFace).
        result = DeepFace.verify(
            img1_path=img1_path, 
            img2_path=img2_path, 
            model_name="ArcFace", 
            detector_backend="mtcnn",
            distance_metric="cosine",
            enforce_detection=False # If no face is found, avoid throwing an error immediately
        )
        
        # DeepFace cosine distance: 0 is identical, higher is different.
        # We convert it to a similarity score where 1.0 is identical.
        distance = result['distance']
        similarity = 1.0 - distance
        
        # Ensure it's between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
        
    except Exception as e:
        print(f"Error in face verification: {e}")
        # Return a neutral or error-indicative score
        return -1.0
