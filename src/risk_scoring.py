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
        
    def calculate_risk(self, mrz_ocr_match, text_alteration_score, face_similarity, ocr_confidence, quality_passed=True):
        """
        Calculates the final Suspicion Score (0 to 100).
        """
        if not quality_passed:
            # Immediate failure / manual review if image cannot be read reliably
            return 95, True

        features = np.array([[mrz_ocr_match, text_alteration_score, face_similarity, ocr_confidence]])
        
        if self.model:
            prob = self.model.predict_proba(features)[0][1]
            score = int(prob * 100)
        else:
            score = 0.0
            
            # 1. MRZ vs OCR Match
            if mrz_ocr_match == 0:
                score += 30
                
            # 2. Text Alteration
            score += (text_alteration_score * 40)
            
            # 3. Face Similarity
            if face_similarity != -1.0:
                if face_similarity < 0.65:
                    score += 50
                else:
                    score += ((1.0 - face_similarity) * 20)
                    
            # 4. OCR Confidence
            score += ((1.0 - ocr_confidence) * 10)
            
            score = min(100, int(score))
            
        flag_manual_review = False
        if face_similarity != -1.0 and face_similarity < 0.65:
            flag_manual_review = True
        if text_alteration_score > 0.8:
            flag_manual_review = True
        if score > 75:
            flag_manual_review = True
            
        return score, flag_manual_review
