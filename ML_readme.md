ML architecture to catch visual anomalies and face inconsistencies from end to end.

**Image Preprocessing Pipeline**
Before any model touches the document, the data must be standardized so the ML doesn't flag bad lighting as a forgery.

- **Deskewing & Cropping:** Use OpenCV to detect document edges (Canny edge detection) and warp the perspective so the document is perfectly flat and aligned.
- **Illumination Correction:** Apply adaptive histogram equalization (CLAHE) to fix shadows, glares, or mobile phone camera artifacts.

**Document Tampering Detection (Visual Anomalies)**
You need a multi-pronged approach here, as no single model catches every type of photoshop or physical alteration.

- **Error Level Analysis (ELA):** When an image is manipulated and resaved, the compressed areas save at different error levels. ELA highlights these discrepancies, making spliced text or swapped photos literally glow in the output.
- **Pixel-Level Forgery Detection:** Implement a pre-trained Convolutional Neural Network (CNN) architecture like Mantra-Net. These are designed specifically to detect copy-move forgeries (copying one digit of a date and pasting it over another) and image splicing.
- **Font Inconsistency:** While PaddleOCR extracts the text strings, you can use a Siamese Neural Network to compare visual patches of the text. If the structural embedding of the font in the "DOB" field drastically differs from the "Name" field, flag it.

**Face Verification (Identity Consistency)**
This step proves the person on the passport is the exact same person on the visa.

- **Face Extraction:** Run MTCNN (Multi-task Cascaded Convolutional Networks) or RetinaFace to detect, align, and crop the face directly from the document.
- **Feature Embedding:** Pass the cropped faces through ArcFace or DeepFace. This generates a high-dimensional vector (an embedding) that maps the exact geometry of the face.
- **Similarity Scoring:** Calculate the Cosine Similarity between the two face embeddings. A score below your set threshold (e.g., < 0.65) strongly indicates a face mismatch.

**Risk Score Aggregation**
Finally, feed your deterministic checks and ML outputs into a centralized scoring model.

| Feature Input             | ML / Logic Source    | Output Type        |
| ------------------------- | -------------------- | ------------------ |
| **MRZ vs. OCR Match**     | Deterministic Script | Boolean (0 or 1)   |
| **Text Alteration Score** | ELA / Forgery CNN    | Float (0.0 to 1.0) |
| **Face Similarity**       | ArcFace (Cosine)     | Float (0.0 to 1.0) |
| **OCR Confidence**        | PaddleOCR Metadata   | Float (0.0 to 1.0) |

Pass these features into a lightweight Gradient Boosting model (like XGBoost) or a weighted algorithm. High-weight failures (like a Face Similarity of 0.2) immediately trigger your manual review flag, while minor OCR uncertainties simply nudge the Suspicion Score up a few points.
