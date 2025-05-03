import asyncio
import logging
import urllib3
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import Optional

from web_crawler.core.crawler import WebCrawler

# --- Logging configuration ---
# Disable some logs from libraries to reduce noise in the output
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.ERROR)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- Global Model Class for Response ---
class CrawlResponse(BaseModel):
    url: HttpUrl
    content: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class HealthCheckOutput(BaseModel):
    """Class to store health check response data."""
    status: str


app = FastAPI(
    title="Web Crawler API",
    description="API to crawl web pages and extract text content.",
    version="1.0.0"
)

log = logging.getLogger("web_crawler_api")

# --- Executor to run Selenium blocking in a separate thread ---
# Helps FastAPI not to be blocked by Selenium
executor = ThreadPoolExecutor()

# --- Function to perform crawl (to run in executor) ---
def run_crawl_sync(target_url: str) -> dict:
    """Synchronous function to perform crawl, returning a result dict."""
    crawler = None
    try:
        log.info(f"Starting crawl for: {target_url}")
        # Initialize the crawler (headless=True is important for server)
        crawler = WebCrawler(headless=True)

        if not crawler.load_page(target_url):
            log.error(f"Failed to load page: {target_url}")
            return {"url": target_url, "error": "Failed to load page"}

        content = crawler.extract_text_content()
        log.info(f"Successfully extracted content (length: {len(content or '')}) for: {target_url}")

        if not content:
            return {"url": target_url, "content": "", "message": "No text content extracted"}
        else:
            return {"url": target_url, "content": content}

    except Exception as e:
        log.error(f"Error during crawl for {target_url}: {e}", exc_info=True)
        return {"url": target_url, "error": f"An error occurred: {str(e)}"}
    finally:
        if crawler:
            log.info(f"Closing crawler for: {target_url}")
            crawler.close()

@app.get("/crawl", response_model=CrawlResponse)
async def crawl_url_endpoint(url: HttpUrl = Query(..., description="URL of the page to crawl")):
    """Receive a URL, use WebCrawler to extract text content and return it."""
    loop = asyncio.get_running_loop()
    try:
        # Run the synchronous crawl function in a ThreadPoolExecutor
        result = await loop.run_in_executor(executor, run_crawl_sync, str(url))

        if "error" in result:
             # If there is an error from the crawl function, return 500 error
             raise HTTPException(status_code=500, detail=result["error"])
        else:
             # Return successful result
             return CrawlResponse(**result)

    except Exception as e:
        # Catch other unexpected errors
        log.error(f"Unexpected API error for {url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected API error occurred: {str(e)}")
    
@app.get("/health", response_model=HealthCheckOutput)
async def health_check():
    """Health check endpoint to verify the service is running."""
    return HealthCheckOutput(status="ready")
