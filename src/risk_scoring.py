import xgboost as xgb
import numpy as np

class RiskScorer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        
        if model_path:
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
            except Exception as e:
                print(f"Could not load XGBoost model from {model_path}. Falling back to weighted algorithm. Error: {e}")
        
    def calculate_risk(self, mrz_ocr_match, text_alteration_score, face_similarity, ocr_confidence):
        """
        Calculates the final Suspicion Score (0 to 100).
        
        Args:
            mrz_ocr_match (int): 0 or 1.
            text_alteration_score (float): 0.0 to 1.0 (Higher means more tampered).
            face_similarity (float): 0.0 to 1.0 (Higher means more similar). -1.0 if not applicable.
            ocr_confidence (float): 0.0 to 1.0.
            
        Returns:
            int: Suspicion score from 0 to 100.
            bool: Flag for manual review.
        """
        features = np.array([[mrz_ocr_match, text_alteration_score, face_similarity, ocr_confidence]])
        
        if self.model:
            # Predict probability of fraud (0.0 to 1.0)
            prob = self.model.predict_proba(features)[0][1]
            score = int(prob * 100)
        else:
            # Fallback: Weighted algorithm
            score = 0.0
            
            # 1. MRZ vs OCR Match (High impact if failed)
            if mrz_ocr_match == 0:
                score += 30
                
            # 2. Text Alteration (High impact)
            score += (text_alteration_score * 40)
            
            # 3. Face Similarity (Very High impact)
            if face_similarity != -1.0:
                if face_similarity < 0.65:
                    # Major penalty for low face similarity
                    score += 50
                else:
                    # Minor penalty for slight variations
                    score += ((1.0 - face_similarity) * 20)
                    
            # 4. OCR Confidence (Low impact)
            score += ((1.0 - ocr_confidence) * 10)
            
            # Normalize to max 100
            score = min(100, int(score))
            
        # Determine if it requires immediate manual review
        flag_manual_review = False
        if face_similarity != -1.0 and face_similarity < 0.65:
            flag_manual_review = True
        if text_alteration_score > 0.8:
            flag_manual_review = True
        if score > 75:
            flag_manual_review = True
            
        return score, flag_manual_review
