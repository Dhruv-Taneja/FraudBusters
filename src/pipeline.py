import os
import cv2

from preprocessing import preprocess_image
from tampering_detection import get_tampering_score
from face_verification import extract_and_verify_faces
from risk_scoring import RiskScorer

from preprocessing import preprocess_image, check_image_quality

def process_documents(doc1_path, doc2_path=None, mrz_match=1, ocr_conf=0.95):
    print(f"--- Starting ML Pipeline for {doc1_path} ---")
    
    # 0. Image Quality Assessment (IQA Gatekeeper)
    raw_img = cv2.imread(doc1_path)
    if raw_img is None:
        print("Error: Could not read image.")
        return 100, True

    is_valid, quality_reason, blur_val = check_image_quality(raw_img, blur_threshold=110.0)
    print(f"0. Quality Check: {quality_reason} (Blur Metric: {blur_val:.2f})")
    
    if not is_valid:
        print("\n--- Final Results ---")
        print(f"Suspicion Score: 95 / 100")
        print("STATUS: FLAGGED FOR MANUAL REVIEW ⚠️ (Reason: Unreadable/Blurry Image)")
        return 95, True

    # 1. Preprocessing
    print("1. Preprocessing image...")
    try:
        preprocessed_img1 = preprocess_image(doc1_path)
        temp_preprocessed = "temp_preprocessed1.jpg"
        cv2.imwrite(temp_preprocessed, preprocessed_img1)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return 100, True
        
    # 2. Tampering Detection
    print("2. Running Tampering Detection...")
    tampering_score = get_tampering_score(temp_preprocessed)
    print(f"   - Text Alteration/Tampering Score: {tampering_score:.2f}")
    
    # 3. Face Verification
    face_sim = -1.0
    if doc2_path:
        print(f"3. Verifying Faces between {doc1_path} and {doc2_path}...")
        face_sim = extract_and_verify_faces(doc1_path, doc2_path)
        print(f"   - Face Similarity Score: {face_sim:.2f}")
    else:
        print("3. Face Verification skipped (only one document provided).")
        
    # 4. Risk Scoring
    print("4. Calculating Risk Score...")
    scorer = RiskScorer()
    score, is_flagged = scorer.calculate_risk(
        mrz_ocr_match=mrz_match, 
        text_alteration_score=tampering_score, 
        face_similarity=face_sim, 
        ocr_confidence=ocr_conf,
        quality_passed=True
    )
    
    print("\n--- Final Results ---")
    print(f"Suspicion Score: {score} / 100")
    if is_flagged:
        print("STATUS: FLAGGED FOR MANUAL REVIEW ⚠️")
    else:
        print("STATUS: PASSED ✅")
        
    if os.path.exists(temp_preprocessed):
        os.remove(temp_preprocessed)
        
    return score, is_flagged

if __name__ == "__main__":
    # Example usage
    # Ensure you have dummy images named 'test_passport.jpg' and 'test_visa.jpg' in the directory to run this locally.
    print("Please provide valid image paths to test the pipeline.")
