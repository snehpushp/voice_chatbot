import os
import time
from pathlib import Path
from typing import Optional

import docx
import requests
from loguru import logger


class Extractor:
    def __init__(self):
        self.llama_api_token = os.getenv("LLAMA_API_TOKEN")
        assert self.llama_api_token, "LLAMA_API_TOKEN is not set. Please keep it in .env file"
        self.base_url = "https://api.cloud.llamaindex.ai/api/v1/parsing"
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.llama_api_token}",
        }

    def extract(self, file_path: str) -> Optional[str]:
        """
        Extract text content from the given file.
        Supports .txt, .md, .docx, and .pdf files.
        """
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".txt":
            return self._extract_txt(file_path)
        elif file_extension == ".md":
            return self._extract_md(file_path)
        elif file_extension == ".docx":
            return self._extract_docx(file_path)
        elif file_extension == ".pdf":
            assert self.llama_api_token, "Llama Cloud API Key is not provided. It is required to parse pdf documents."
            return self._extract_pdf(file_path)
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return None

    @staticmethod
    def _extract_txt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _extract_md(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            return file.read()  # Return raw markdown

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _extract_pdf(self, file_path: str) -> Optional[str]:
        # Upload PDF
        upload_url = f"{self.base_url}/upload"
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f)}
            response = requests.post(upload_url, headers=self.headers, files=files)
            if response.status_code != 200:
                logger.error(f"Failed to upload PDF: {response.text}")
                return None
            upload_result = response.json()

        job_id = upload_result["id"]

        # Check job status
        while True:
            status_url = f"{self.base_url}/job/{job_id}"
            response = requests.get(status_url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Failed to check job status: {response.text}")
                return None
            status_result = response.json()

            if status_result["status"] == "SUCCESS":
                break
            elif status_result["status"] in ["ERROR", "PARTIAL_SUCCESS"]:
                logger.error(f"Job failed: {status_result.get('error_message', 'Unknown error')}")
                return None

            time.sleep(2)  # Wait before checking again

        # Get markdown result
        result_url = f"{self.base_url}/job/{job_id}/result/raw/markdown"
        response = requests.get(result_url, headers=self.headers)
        if response.status_code != 200:
            logger.error(f"Failed to get markdown result: {response.text}")
            return None
        markdown_content = response.text

        return markdown_content


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    def main():
        # Example usage
        extractor = Extractor()

        file_paths = [
            "input_text.txt",
            r"C:\Users\pushp\Downloads\UPSC Books\PYQs\Mains\UPSC GS 3 PYQ ANALYSIS.pdf",
            r"C:\Users\pushp\Downloads\Posters.docx",
        ]

        for file_path in file_paths:
            content = extractor.extract(file_path)
            if content:
                logger.info(f"Successfully extracted content from {file_path}")
                logger.info(f"First 100 characters: {content[:100]}...")
            else:
                logger.error(f"Failed to extract content from {file_path}")

    main()
