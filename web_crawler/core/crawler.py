"""
Core crawler functionality for extracting content from websites.
"""


import os
import time
import urllib3

import certifi
import re
import requests
import ssl
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import validators

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from web_crawler.utils.log import get_logger

# Disable SSL verification for WebDriver Manager
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["WDM_SSL_VERIFY"] = "0"

# Configure logger
log = get_logger("web_crawler")


class WebCrawler:
    """
    WebCrawler class that handles the entire crawling process.
    Combines both browser automation and content extraction.
    """

    def __init__(self, headless=True):
        """
        Initialize the WebCrawler with a Selenium WebDriver.

        Args:
            headless (bool): Run browser in headless mode if True.
        """
        self.driver = None
        self.headless = headless
        self._initialize_driver()

        # Disable SSL verification for requests
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Set default SSL context to unverified
        ssl._create_default_https_context = ssl._create_unverified_context

    def __del__(self):
        """Ensure browser is closed when object is destroyed."""
        self.close()

    def _initialize_driver(self):
        """Set up Chrome WebDriver with optimized options."""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Add performance options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.driver.set_page_load_timeout(30)  # 30 seconds timeout
            log.info("WebDriver initialized successfully")
        except Exception as e:
            log.error(f"Error initializing WebDriver: {e}")
            raise

    def close(self):
        """Close the WebDriver if it exists."""
        if self.driver:
            try:
                self.driver.quit()
                log.info("WebDriver closed successfully")
            except Exception as e:
                log.error(f"Error closing WebDriver: {e}")
            self.driver = None

    def load_page(self, url):
        """
        Load a web page in the browser.

        Args:
            url (str): URL of the page to load.
            
        Returns:
            bool: True if page loaded successfully, False otherwise.
        """
        try:
            log.info(f"Loading page: {url}")
            self.driver.get(url)
            # Wait for page to load
            time.sleep(5)
            return True
        except Exception as e:
            log.error(f"Error loading page {url}: {e}")
            return False

    def extract_links(self):
        """
        Extract all links from current page.
        
        Returns:
            list: List of URLs found on the page.
        """
        try:
            log.debug("Extracting links from current page")
            self.scroll_to_bottom()
            links = self.driver.find_elements(By.TAG_NAME, "a")
            urls = []
            
            for link in links:
                href = link.get_attribute("href")
            if href:
                absolute_url = urljoin(self.driver.current_url, href)
                urls.append(absolute_url)
            
            log.info(f"Found {len(urls)} links")
            return urls
        except Exception as e:
            log.error(f"Error extracting links: {e}")
            return []

    def scroll_to_bottom(self):
        """
        Scroll to the bottom of the page to load dynamic content.
        """
        try:
            # Get scroll height
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while True:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                # Wait to load page
                time.sleep(1)
                
                # Calculate new scroll height and compare with last scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
        except Exception as e:
            log.error(f"Error scrolling: {e}")

    def extract_text_content(self):
        """
        Extract plain text content from the current page.

        Returns:
            str: Text content of the page.
        """
        try:
            self.scroll_to_bottom()  # Ensure all content is loaded

            # Find the main content element (adjust selector if needed)
            try:
                content_element = self.driver.find_element(By.CSS_SELECTOR, "div.item-page")
                text = content_element.text
            except Exception as find_err:
                log.warning(f"🚨 Could not find specific content element (div.item-page), falling back to body.innerText")
                # Fallback to body.innerText if specific element not found
                js_code = "return document.body.innerText;"
                text = self.driver.execute_script(js_code)
            return text
        except Exception as e:
            log.error(f"🚨🚨 Error extracting text content: {e}")
            return ""

    def extract_domain(self, url):
        """
        Extract the main domain from a given URL.

        Args:
            url (str): The URL to parse.
            
        Returns:
            str: The domain name (e.g., "www.example.com").
        """
        return urlparse(url).netloc

    def crawl_links(self, base_url, max_depth=3, visited=None):
        """
        Recursively crawl links from a base URL up to a specified depth.

        Args:
            base_url (str): The starting URL for crawling.
            max_depth (int): Maximum depth for recursive crawling.
            visited (set): Set of visited URLs to avoid duplicates.
            
        Returns:
            list: List of unique URLs crawled.
        """
        if max_depth == 0:
            return []
        if visited is None:
            visited = set()
        if base_url in visited:
            return []
            
        log.info(f"Crawling: {base_url} (depth: {max_depth})")
        visited.add(base_url)

        if not self.load_page(base_url):
            return []
            
        all_urls = self.extract_links()

        # Filter URLs to include only those from the same domain
        base_domain = self.extract_domain(base_url)
        filtered_urls = [
            url for url in all_urls
            if url and validators.url(url)
            and self.extract_domain(url) == base_domain
        ]
        
        # Filter out URLs with patterns like /yyyy/mm/dd/
        crawled_urls = self.filter_internal_urls(filtered_urls)
        
        log.info(f"Found {len(crawled_urls)} relevant URLs at depth {max_depth}")
        
        # Recursively crawl child links
        if max_depth > 1:
            for url in filtered_urls:
                if url not in visited:
                    child_urls = self.crawl_links(url, max_depth - 1, visited)
                    crawled_urls.extend(child_urls)
        
        return list(set(crawled_urls))

    def filter_internal_urls(self, urls):
        """
        Filter out URLs that match certain patterns.
        
        Args:
            urls (list): List of URLs to filter.
            
        Returns:
            list: Filtered list of URLs.
        """
        # Filter out URLs with patterns like /yyyy/mm/dd/
        pattern = re.compile(r"https?://[^/]+/\d{4}/\d{2}/\d{2}/")
        return [url for url in urls if not pattern.search(url)]

    def download_images(self, url, output_dir):
        """
        Download all images from a webpage.

        Args:
            url (str): The URL of the webpage to scrape images from.
            output_dir (str): Directory to save downloaded images.
            
        Returns:
            dict: Dictionary mapping image URLs to their local file paths.
        """
        if not self.load_page(url):
            return {}
            
        log.info(f"Downloading images from {url}")
        
        images = self.driver.find_elements(By.TAG_NAME, "img")
        log.info(f"Found {len(images)} images")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_dict = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, img in enumerate(images):
                img_url = img.get_attribute("src")
                if img_url and not img_url.startswith("data:"):
                    img_name = f"{i}_{os.path.basename(urlparse(img_url).path)}"
                    if not img_name or img_name == f"{i}_":
                        img_name = f"{i}_image.jpg"
                    img_path = os.path.join(output_dir, img_name)
                    futures.append(executor.submit(self._download_image, img_url, img_path))
            
            for future in as_completed(futures):
                img_url, img_path = future.result()
                if os.path.exists(img_path):
                    img_dict[img_url] = img_path

        log.info(f"Downloaded {len(img_dict)} images")
        return img_dict

    @staticmethod
    def _download_image(img_url, img_path):
        """
        Download an image from a URL and save it locally.

        Args:
            img_url (str): URL of the image to download.
            img_path (str): Local path to save the image.
            
        Returns:
            tuple: Tuple of (image URL, local file path).
        """
        try:
            response = requests.get(img_url, stream=True, verify=False, timeout=10)
            if response.status_code == 200:
                # Handle webp images
                if img_url.lower().endswith(".webp"):
                    img_path = os.path.splitext(img_path)[0] + ".jpg"
                    
                with open(img_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                        
                return img_url, img_path
            return img_url, None
        except Exception as e:
            log.error(f"Error downloading image {img_url}: {e}")
            return img_url, None

    def crawl_website(self, base_url, output_dir, max_depth=3, download_images=False):
        """
        Main method to crawl a website, extract links and content.
        
        Args:
            base_url (str): The starting URL to crawl.
            output_dir (str): Directory to save crawled data.
            max_depth (int): Maximum depth for recursive crawling.
            download_images (bool): Whether to download images.
            
        Returns:
            tuple: (list of crawled URLs, dictionary of page contents)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Crawl all links (optional, depending on depth)
        found_links = []
        if max_depth > 0:
            found_links = self.crawl_links(base_url, max_depth)
            log.info(f"Crawled {len(found_links)} links starting from {base_url}")
        else:
             log.info(f"max_depth is 0, only processing the base_url: {base_url}")


        # Ensure the base_url itself is always processed for content
        urls_to_process = list(set([base_url] + found_links))
        log.info(f"Total unique URLs to process for content: {len(urls_to_process)}")
        
        # Step 2: Extract content from each page in the combined list
        page_contents = {}
        for i, link in enumerate(urls_to_process): 
            log.info(f"Processing page {i+1}/{len(urls_to_process)}: {link}")
            if self.load_page(link):
                content = self.extract_text_content()
                if content:
                    page_contents[link] = content
        
        # Step 3: Download images if requested (only from base_url for simplicity now)
        if download_images:
            images_dir = os.path.join(output_dir, "images")
            log.info(f"Attempting to download images from base URL: {base_url}")
            self.download_images(base_url, images_dir)
        
        # Return the list of links found during crawling (excluding base_url unless found) and the extracted contents
        return found_links, page_contents