from paddleocr import PaddleOCR
import re

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def extract_document_data(image_path):
    result = ocr.predict(image_path)
    res = result[0]

    texts = res["rec_texts"]
    scores = res["rec_scores"]

    GOOD_THRESHOLD = 0.85
    recorded_texts = []

    date_pattern = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")

    for text, score in zip(texts, scores):
        if float(score) < GOOD_THRESHOLD:
            continue

        text = text.strip()

        if "/" in text and not date_pattern.fullmatch(text):
            text = text.split("/", 1)[1].strip()

        if text:
            recorded_texts.append(text)

    return recorded_texts