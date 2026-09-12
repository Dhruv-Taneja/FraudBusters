
**Document Upload → OCR Extraction → Document Validation → Tampering Detection → Face Verification → Risk Score**

We will be handling the document verification part. The user will upload documents like a passport, visa, etc. Initially, the user will tell us what type of document it is, like Passport, Visa, etc.

Then we use **PaddleOCR**, which is a Python library, to get the actual data from the documents in the form of key-value pairs, like:

Name: ABC
DOB: XYZ
Passport Number: XYZ

We need to check PaddleOCR once to understand exactly how it works.

For passports, we can also use that weird **MRZ** section:

P<INDSHARMA<<RAHUL<<<<<<<<<<<<<<<<<<<<

A1234567<8IND0208159M3208145<<<<<<<<<<

It also contains important data, so we need to see how we can extract data from it. We can compare the normal document data and MRZ data, which can also work as one basic fraud check. But the main purpose of OCR is to get the data from the documents.

### Document Validation

Every document follows a particular structure. For example, a passport stores data differently compared to Aadhaar, and each type of data also has different rules.

For example:

* Date has different validation rules.
* Name has different validation rules.
* Passport number has a particular format.

We validate the data according to the document type and then compare the same information across all uploaded documents to check consistency.

### Tampering Detection

Apart from document validation, we will use ML for each document to look for inconsistencies.

For example:

* Different font or font inconsistencies.
* Text manipulation.
* Other visual anomalies.
* Photo being significantly different across documents.

Basically, ML will analyze each document and try to find any suspicious visual changes or anomalies.

### Risk Score and Flagging

Based on all the above non-ML checks, if something important fails, we flag it immediately.

By **flag**, I mean we notify the authorities that this document requires actual human verification.

Otherwise, we generate a prediction percentage:

**Suspicion Score:**

0 → No suspicion
100 → Extremely suspicious

This way, authorities don't have to manually verify every document. They can focus more on documents that are flagged or have a high suspicion score, reducing their overall workload.