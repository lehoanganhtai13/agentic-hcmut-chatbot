"""
Data processing utilities for the web crawler.
Handles saving data to various formats.
"""

import os
import re
import csv
import json
from langdetect import detect, LangDetectException
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from web_crawler.utils.log import get_logger

# Configure Log
log = get_logger("web_crawler")


class DataProcessor:
    """
    Handles processing and saving of crawled data to various formats.
    """

    def __init__(self, font_path: Optional[str] = None):
        """
        Initialize font settings for PDF generation.
        
        Args:
            font_path (Optional[str]): Path to custom font for PDF.
        """
        self.font_name = "Helvetica"  # Default font
        
        if PDF_SUPPORT:
            try:
                # Try to use Times New Roman if available
                if font_path and os.path.exists(font_path):
                    self.font_name = font_path.split("/")[-1].replace(".ttf", "").replace(" ", "_")
                    pdfmetrics.registerFont(TTFont(self.font_name, font_path))
                    log.info(f"Using font {self.font_name.replace('_', ' ').title()}")
                else:
                    log.info("Using default Helvetica font")
            except Exception as e:
                log.error(f"Error loading fonts: {e}")

    def save_to_json(self, file_path: str, data: Dict) -> bool:
        """
        Save data to a JSON file.
        
        Args:
            file_path (str): Path to save the JSON file.
            data (Dict): Data to save.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            log.info(f"Data saved to JSON: {file_path}")
            return True
        except Exception as e:
            log.error(f"Error saving to JSON: {e}")
            return False

    def save_to_csv(self, file_path: str, data: List[Dict], headers: List[str]) -> bool:
        """
        Save data to a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            data (List[Dict]): List of dictionaries representing rows of data.
            headers (List[str]): List of column headers.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file_exists = os.path.isfile(file_path)
            with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(data)
                
            log.info(f"Data saved to CSV: {file_path}")
            return True
        except Exception as e:
            log.error(f"Error saving to CSV: {e}")
            return False

    def update_csv_status(self, file_path: str, row_id: str, status: str) -> bool:
        """
        Update the STATUS field of a specific row in the CSV file.

        Args:
            file_path (str): Path to the CSV file.
            row_id (str): ID of the row to update.
            status (str): New status value (e.g., "COMPLETED").
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            updated = False
            for row in rows:
                if str(row.get("ID")) == str(row_id):
                    row["STATUS"] = status
                    updated = True
                    break

            if not updated:
                log.warning(f"Row ID {row_id} not found in {file_path}")
                return False

            with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
            log.info(f"Updated row {row_id} status to {status} in {file_path}")
            return True
        except Exception as e:
            log.error(f"Error updating CSV: {e}")
            return False

    def save_text_as_pdf(self, text: str, output_pdf_path: str) -> bool:
        """
        Save extracted text as a formatted PDF file.

        Args:
            text (str): Text content to save.
            output_pdf_path (str): Path to the output PDF file.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not PDF_SUPPORT:
            log.error("PDF support not available. Install reportlab package.")
            return False
            
        try:
            output_dir = os.path.dirname(output_pdf_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            custom_style = ParagraphStyle(
                name="CustomStyle", 
                fontName=self.font_name, 
                fontSize=12, 
                leading=14
            )
            
            # Split text into paragraphs and create flowables
            flowables = [
                Paragraph(escape(paragraph), custom_style) 
                for paragraph in text.split("\n") if paragraph.strip()
            ]
            
            doc.build(flowables)
            log.info(f"Created PDF: {output_pdf_path}")
            return True
        except Exception as e:
            log.error(f"Error creating PDF: {e}")
            return False

    def save_text_as_txt(self, text: str, output_txt_path: str) -> bool:
        """
        Save extracted text as a plain text file.

        Args:
            text (str): Text content to save.
            output_txt_path (str): Path to the output text file.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            output_dir = os.path.dirname(output_txt_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            log.info(f"Created text file: {output_txt_path}")
            return True
        except Exception as e:
            log.error(f"Error creating text file: {e}")
            return False

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text.
        
        Args:
            text (str): Text to analyze.
            
        Returns:
            str: Language code (e.g., "en", "vi") or "unknown".
        """
        try:
            if not text or len(text.strip()) < 20:
                return "unknown"
            return detect(text)
        except LangDetectException:
            return "unknown"

    def extract_domain_sublink(self, url: str) -> str:
        """
        Extract the last part of the URL path (sublink).

        Args:
            url (str): The URL to parse.
            
        Returns: 
            str: The sublink or None if not found.
        """
        pattern = r"[^/]+(?=/$|$)"
        match = re.search(pattern, url)
        return match.group() if match else None

    def create_url_list_csv(self, folder_name: str, list_urls: List[str]):
        """
        Save a list of URLs to a CSV file with metadata.

        Args:
            folder_name (str): Name of the folder to store the CSV file.
            list_urls (List[str]): List of URLs to save.
            
        Returns: 
            str: Path to the saved CSV file.
        """
        try:
            file_path = os.path.join("data", folder_name, f"{folder_name}.csv")
            output_dir = os.path.dirname(file_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            log.info(f"Saving links to CSV file: {file_path}")

            rows = []
            for idx, url in enumerate(list_urls, start=1):
                file_name = f"data_{idx}.pdf"
                title = self.extract_domain_sublink(url)
                if title:
                    title = title.replace("-", " ")
                else:
                    title = "Untitled"
                    
                rows.append({
                    "ID": idx,
                    "URL": url,
                    "FILENAME": file_name,
                    "TITLE": title,
                    "STATUS": "PENDING"
                })
                
            headers = ["ID", "URL", "FILENAME", "TITLE", "STATUS"]
            self.save_to_csv(file_path, rows, headers)
            
            log.info("CSV file created successfully")
            return file_path
        except Exception as e:
            log.error(f"Error creating URL list CSV: {e}")
            return None

    def process_and_save_data(self, urls_with_content, output_dir, file_format="txt"):
        """
        Process and save multiple pages of content.
        
        Args:
            urls_with_content (dict): Dictionary mapping URLs to their content.
            output_dir (str): Directory to save processed content.
            file_format (str): Format to save content ("txt" or "pdf").
            
        Returns:
            dict: Dictionary with results for each URL.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Create CSV file with URLs
        csv_path = self.create_url_list_csv(
            os.path.basename(output_dir), 
            list(urls_with_content.keys())
        )
        
        for idx, (url, content) in enumerate(urls_with_content.items(), start=1):
            try:
                # Detect language
                lang = self.detect_language(content)
                
                # Create file path
                if file_format.lower() == "pdf":
                    file_path = os.path.join(output_dir, f"data_{idx}.pdf")
                    success = self.save_text_as_pdf(content, file_path)
                else:
                    file_path = os.path.join(output_dir, f"data_{idx}.txt")
                    success = self.save_text_as_txt(content, file_path)
                
                if success and csv_path:
                    self.update_csv_status(csv_path, str(idx), "COMPLETED")
                
                results[url] = {
                    "file": file_path,
                    "language": lang,
                    "success": success
                }
            except Exception as e:
                log.error(f"Error processing URL {url}: {e}")
                results[url] = {
                    "file": None,
                    "language": "unknown",
                    "success": False,
                    "error": str(e)
                }
                
        return results