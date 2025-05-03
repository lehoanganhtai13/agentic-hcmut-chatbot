"""
Web Crawler package for extracting data from websites.
"""

from web_crawler.core.crawler import WebCrawler
from web_crawler.core.processor import DataProcessor


__all__ = [
    "WebCrawler",
    "DataProcessor"
]