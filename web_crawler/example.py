"""
Example usage of the web_crawler module.
"""

from web_crawler.core.crawler import WebCrawler
from web_crawler.core.processor import DataProcessor

def main():
    # Initialize crawler & processor
    crawler = WebCrawler(headless=True)
    processor = DataProcessor(font_path="web_crawler/fonts/Times New Roman.ttf")
    
    # Define target website and output directory
    url = "https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Vector/delete.md"
    output_dir = "crawled_data"
    
    try:
        # Crawl website and get contents
        links, contents = crawler.crawl_website(
            base_url=url,
            output_dir=output_dir,
            max_depth=1,
            download_images=True
        )
        
        # Process and save the crawled data
        results = processor.process_and_save_data(
            urls_with_content=contents,
            output_dir=output_dir,
            file_format="txt"  # or "pdf"
        )
        
        # Print results
        print(f"Crawled {len(links)} links")
        print(f"Successfully processed {sum(1 for r in results.values() if r['success'])} pages")
        
    finally:
        # Always close the crawler
        crawler.close()

if __name__ == "__main__":
    main()