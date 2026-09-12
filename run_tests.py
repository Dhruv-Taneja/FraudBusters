import cv2
import numpy as np
import sys
import os

# Add src to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_image
from tampering_detection import get_tampering_score, perform_ela
from face_verification import extract_and_verify_faces
from risk_scoring import RiskScorer
from pipeline import process_documents

def create_dummy_image(path, text="Dummy Document"):
    """Helper to create a dummy image if you don't have one."""
    img = np.ones((500, 800, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imwrite(path, img)
    return path

def test_preprocessing():
    print("\n--- Testing Preprocessing ---")
    img_path = create_dummy_image("dummy_doc.jpg")
    try:
        processed = preprocess_image(img_path)
        print("Preprocessing successful! Output shape:", processed.shape)
        cv2.imwrite("dummy_preprocessed.jpg", processed)
        print("Saved preprocessed image to dummy_preprocessed.jpg")
    except Exception as e:
        print("Preprocessing failed:", e)

def test_tampering():
    print("\n--- Testing Tampering Detection (ELA) ---")
    img_path = "dummy_doc.jpg"
    if not os.path.exists(img_path):
        create_dummy_image(img_path)
        
    score, ela_img = perform_ela(img_path)
    print(f"ELA Anomaly Score: {score:.4f}")
    ela_img.save("dummy_ela_result.jpg")
    print("Saved ELA visualization to dummy_ela_result.jpg")

def test_risk_scorer():
    print("\n--- Testing Risk Scorer ---")
    scorer = RiskScorer()
    # Scenario 1: Clean document
    score, flag = scorer.calculate_risk(mrz_ocr_match=1, text_alteration_score=0.1, face_similarity=0.9, ocr_confidence=0.95)
    print(f"Clean Doc -> Score: {score}, Flagged: {flag}")
    
    # Scenario 2: Fraudulent document (Face mismatch)
    score, flag = scorer.calculate_risk(mrz_ocr_match=1, text_alteration_score=0.2, face_similarity=0.4, ocr_confidence=0.90)
    print(f"Bad Face Match Doc -> Score: {score}, Flagged: {flag}")

def test_full_pipeline():
    print("\n--- Testing Full Pipeline ---")
    doc1 = create_dummy_image("dummy_passport.jpg", "Passport")
    doc2 = create_dummy_image("dummy_visa.jpg", "Visa")
    
    # Note: Face verification will fail gracefully since there are no faces in the dummy images.
    process_documents(doc1_path=doc1, doc2_path=doc2, mrz_match=1, ocr_conf=0.92)

if __name__ == "__main__":
    print("Ensure you have run: pip install -r requirements.txt")
    test_preprocessing()
    test_tampering()
    test_risk_scorer()
    test_full_pipeline()
