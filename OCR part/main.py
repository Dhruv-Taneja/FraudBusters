import asyncio
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI

from ocr_part import extract_document_data
from verified import verify_document_data


app = FastAPI()


NODE_SERVER_URL = "http://localhost:3000/document"

DOWNLOAD_FOLDER = Path("downloads")
DOWNLOAD_FOLDER.mkdir(exist_ok=True)


# Results waiting for Node.js server 2
manual_verification_results = []


def get_file_extension(document_url: str) -> str:
    """
    Detect file extension from the document URL.
    """

    url_without_query = document_url.split("?")[0].lower()

    if url_without_query.endswith(".pdf"):
        return ".pdf"

    if url_without_query.endswith(".png"):
        return ".png"

    if url_without_query.endswith(".jpeg"):
        return ".jpeg"

    if url_without_query.endswith(".webp"):
        return ".webp"

    # Default extension
    return ".jpg"


async def download_document(
    client: httpx.AsyncClient,
    document_url: str
) -> Path:

    extension = get_file_extension(document_url)

    file_name = f"{uuid.uuid4().hex}{extension}"
    file_path = DOWNLOAD_FOLDER / file_name

    print("Downloading document from:", document_url)

    response = await client.get(document_url)
    response.raise_for_status()

    file_path.write_bytes(response.content)

    print("Document saved at:", file_path)

    return file_path


async def poll_node_server():

    async with httpx.AsyncClient(timeout=30) as client:

        while True:

            try:
                response = await client.get(NODE_SERVER_URL)
                response.raise_for_status()

                data = response.json()

                if not data.get("available"):
                    print("No document available")

                else:
                    document_id = data.get("document_id")
                    document_type = data.get("document_type")
                    document_url = data.get("url")

                    if not document_id:
                        print("Missing document_id")

                    elif not document_type:
                        print("Missing document_type")

                    elif not document_url:
                        print("Missing document URL")

                    else:
                        print("New document received:", document_id)
                        print("Document type:", document_type)

                        file_path = None

                        try:
                            # 1. Download document
                            file_path = await download_document(
                                client,
                                document_url
                            )

                            # 2. OCR
                            extracted_data = await asyncio.to_thread(
                                extract_document_data,
                                str(file_path)
                            )

                            print("OCR completed")

                            # 3. Verification
                            # document_type comes from the JavaScript server
                            verification_result = await asyncio.to_thread(
                                verify_document_data,
                                extracted_data,
                                document_type
                            )

                            print("Verification completed")

                            # 4. Check manual verification
                            needs_manual_verification = (
                                verification_result.get(
                                    "needs_manual_verification",
                                    True
                                )
                            )

                            if needs_manual_verification:

                                result = {
                                    "document_id": document_id,
                                    "document_type": document_type,
                                    "ocr_data": extracted_data,
                                    "verification_result": verification_result
                                }

                                manual_verification_results.append(result)

                                print(
                                    "Result stored for manual verification:",
                                    document_id
                                )

                            else:
                                print(
                                    "Document verified automatically:",
                                    document_id
                                )

                        finally:
                            # Delete temporary file
                            if file_path is not None:
                                file_path.unlink(
                                    missing_ok=True
                                )

            except httpx.HTTPError as error:
                print(
                    "Node.js or document download error:",
                    error
                )

            except Exception as error:
                print(
                    "Unexpected error:",
                    error
                )

            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():

    asyncio.create_task(
        poll_node_server()
    )


@app.get("/")
async def home():

    return {
        "message": "FastAPI document polling server is running"
    }


@app.get("/verification-result")
async def get_verification_result():

    # No result is currently waiting
    if not manual_verification_results:
        return 0

    # Copy currently waiting results
    results = manual_verification_results.copy()

    # Remove them after sending them to Node.js server 2
    manual_verification_results.clear()

    return {
        "count": len(results),
        "documents": results
    }