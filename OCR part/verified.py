import re


# ---------------------------------------------------------
# COMMON HELPERS
# ---------------------------------------------------------

def clean_text(text):
    """
    Normalize OCR text.
    """
    text = str(text).strip()
    text = text.replace("≤", "<")
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_texts(extracted_data):
    """
    Convert OCR output into clean text lines.
    """
    return [
        clean_text(text)
        for text in extracted_data
        if clean_text(text)
    ]


def find_text_after_label(texts, labels):
    """
    Finds the text immediately after a label.

    Example:
    ['Name', 'RAHUL KUMAR']
    labels = ['name']
    returns 'RAHUL KUMAR'
    """
    labels = [label.lower() for label in labels]

    for index, text in enumerate(texts):
        current = text.lower().strip()

        if current in labels:
            if index + 1 < len(texts):
                return texts[index + 1]

    return None


def contains_any(text, keywords):
    """
    Checks whether any keyword exists in text.
    """
    text = text.lower()

    return any(keyword.lower() in text for keyword in keywords)


# ---------------------------------------------------------
# PASSPORT VERIFICATION
# ---------------------------------------------------------

def verify_passport(extracted_data):
    texts = normalize_texts(extracted_data)

    full_text = " ".join(texts).upper()

    passport_number = None
    surname = None
    given_names = None
    sex = None
    date_of_birth = None
    place_of_birth = None
    place_of_issue = None
    mrz_lines = []

    # Passport number
    passport_pattern = re.compile(r"\b[A-Z][0-9]{7}\b")

    for text in texts:
        match = passport_pattern.search(text.upper())

        if match:
            passport_number = match.group()
            break

    # Surname
    surname = find_text_after_label(
        texts,
        ["surname", "sur name"]
    )

    # Given names
    given_names = find_text_after_label(
        texts,
        ["given names", "given name", "name"]
    )

    # Sex
    sex = find_text_after_label(
        texts,
        ["sex"]
    )

    # Date of birth
    date_pattern = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    )

    for index, text in enumerate(texts):
        if contains_any(text, ["date of birth", "dob"]):
            if index + 1 < len(texts):
                possible_date = texts[index + 1]
                match = date_pattern.search(possible_date)

                if match:
                    date_of_birth = match.group()
                    break

    # Place of birth
    place_of_birth = find_text_after_label(
        texts,
        ["place of birth"]
    )

    # Place of issue
    place_of_issue = find_text_after_label(
        texts,
        ["place of issue"]
    )

    # MRZ lines
    for text in texts:
        upper_text = text.upper()

        if upper_text.startswith("P<") or upper_text.startswith("P<IND"):
            mrz_lines.append(upper_text)

        elif re.fullmatch(r"[A-Z0-9<]{30,}", upper_text):
            mrz_lines.append(upper_text)

    missing_fields = []

    if not passport_number:
        missing_fields.append("passport_number")

    if not surname:
        missing_fields.append("surname")

    if not date_of_birth:
        missing_fields.append("date_of_birth")

    result = {
        "document_type": "passport",
        "is_document_format_valid": bool(passport_number),
        "passport_number": passport_number,
        "surname": surname,
        "given_names": given_names,
        "sex": sex,
        "date_of_birth": date_of_birth,
        "place_of_birth": place_of_birth,
        "place_of_issue": place_of_issue,
        "mrz_lines": mrz_lines,
        "missing_fields": missing_fields,
        "needs_manual_verification": len(missing_fields) > 0,
        "message": (
            "Passport OCR extraction successful"
            if len(missing_fields) == 0
            else "Some passport fields are missing"
        )
    }

    return result


# ---------------------------------------------------------
# AADHAAR VERIFICATION
# ---------------------------------------------------------

def verify_aadhaar(extracted_data):
    texts = normalize_texts(extracted_data)

    full_text = " ".join(texts).upper()

    aadhaar_number = None
    name = None
    date_of_birth = None
    year_of_birth = None
    gender = None
    address = None

    # Aadhaar number formats:
    # 1234 5678 9012
    # 123456789012
    aadhaar_pattern = re.compile(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    )

    for text in texts:
        match = aadhaar_pattern.search(text)

        if match:
            aadhaar_number = re.sub(
                r"\s+",
                "",
                match.group()
            )
            break

    # Name
    name = find_text_after_label(
        texts,
        ["name", "नाम"]
    )

    # Date of birth
    date_pattern = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    )

    for index, text in enumerate(texts):
        if contains_any(
            text,
            [
                "date of birth",
                "dob",
                "जन्म तिथि"
            ]
        ):
            if index + 1 < len(texts):
                possible_date = texts[index + 1]
                match = date_pattern.search(possible_date)

                if match:
                    date_of_birth = match.group()
                    break

    # Year of birth
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")

    for index, text in enumerate(texts):
        if contains_any(
            text,
            [
                "year of birth",
                "yob",
                "जन्म वर्ष"
            ]
        ):
            if index + 1 < len(texts):
                match = year_pattern.search(texts[index + 1])

                if match:
                    year_of_birth = match.group()
                    break

    # Gender
    for text in texts:
        upper_text = text.upper()

        if upper_text in ["MALE", "पुरुष"]:
            gender = "Male"
            break

        if upper_text in ["FEMALE", "महिला"]:
            gender = "Female"
            break

        if upper_text in ["TRANSGENDER", "अन्य"]:
            gender = "Transgender"
            break

    # Address
    address_start = None

    for index, text in enumerate(texts):
        if contains_any(
            text,
            ["address", "पता"]
        ):
            address_start = index + 1
            break

    if address_start is not None:
        address_parts = []

        for text in texts[address_start:]:
            if contains_any(
                text,
                [
                    "government of india",
                    "unique identification",
                    "uidai",
                    "www.uidai.gov.in"
                ]
            ):
                continue

            address_parts.append(text)

        if address_parts:
            address = ", ".join(address_parts)

    missing_fields = []

    if not aadhaar_number:
        missing_fields.append("aadhaar_number")

    if not name:
        missing_fields.append("name")

    if not date_of_birth and not year_of_birth:
        missing_fields.append("date_of_birth_or_year_of_birth")

    result = {
        "document_type": "aadhaar",
        "is_document_format_valid": bool(aadhaar_number),
        "aadhaar_number": aadhaar_number,
        "name": name,
        "date_of_birth": date_of_birth,
        "year_of_birth": year_of_birth,
        "gender": gender,
        "address": address,
        "missing_fields": missing_fields,
        "needs_manual_verification": len(missing_fields) > 0,
        "message": (
            "Aadhaar OCR extraction successful"
            if len(missing_fields) == 0
            else "Some Aadhaar fields are missing"
        )
    }

    return result


# ---------------------------------------------------------
# PAN CARD VERIFICATION
# ---------------------------------------------------------

def verify_pan(extracted_data):
    texts = normalize_texts(extracted_data)

    full_text = " ".join(texts).upper()

    pan_number = None
    name = None
    father_name = None
    date_of_birth = None

    # PAN format:
    # Five letters + four digits + one letter
    #
    # Example:
    # ABCDE1234F
    pan_pattern = re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    )

    for text in texts:
        match = pan_pattern.search(text.upper())

        if match:
            pan_number = match.group()
            break

    # Name
    name = find_text_after_label(
        texts,
        ["name", "नाम"]
    )

    # Father's name
    father_name = find_text_after_label(
        texts,
        [
            "father's name",
            "father name",
            "father",
            "पिता का नाम"
        ]
    )

    # Date of birth
    date_pattern = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    )

    for index, text in enumerate(texts):
        if contains_any(
            text,
            [
                "date of birth",
                "dob",
                "date of incorporation",
                "date of formation",
                "जन्म तिथि"
            ]
        ):
            if index + 1 < len(texts):
                possible_date = texts[index + 1]
                match = date_pattern.search(possible_date)

                if match:
                    date_of_birth = match.group()
                    break

    missing_fields = []

    if not pan_number:
        missing_fields.append("pan_number")

    if not name:
        missing_fields.append("name")

    if not date_of_birth:
        missing_fields.append("date_of_birth")

    result = {
        "document_type": "pan",
        "is_document_format_valid": bool(pan_number),
        "pan_number": pan_number,
        "name": name,
        "father_name": father_name,
        "date_of_birth": date_of_birth,
        "missing_fields": missing_fields,
        "needs_manual_verification": len(missing_fields) > 0,
        "message": (
            "PAN OCR extraction successful"
            if len(missing_fields) == 0
            else "Some PAN fields are missing"
        )
    }

    return result


# ---------------------------------------------------------
# MAIN DISPATCHER
# ---------------------------------------------------------

def verify_document_data(extracted_data, document_type):
    """
    Main function called from main.py.
    """

    if not document_type:
        return {
            "document_type": None,
            "needs_manual_verification": True,
            "message": "Document type was not provided"
        }

    document_type = document_type.lower().strip()

    if document_type == "passport":
        return verify_passport(extracted_data)

    elif document_type in ["aadhaar", "aadhar"]:
        return verify_aadhaar(extracted_data)

    elif document_type == "pan":
        return verify_pan(extracted_data)

    else:
        return {
            "document_type": document_type,
            "needs_manual_verification": True,
            "message": "Unsupported document type"
        }