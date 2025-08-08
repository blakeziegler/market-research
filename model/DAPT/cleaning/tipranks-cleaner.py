import requests
from bs4 import BeautifulSoup
import re
import time
from typing import Dict, List, Optional
import logging
import os

# Try to import Selenium for browser automation
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Install with: pip install selenium")

class TipRanksCleaner:
    """
    Streamlined HTML parser for TipRanks financial data extraction.
    Uses minimal selectors to extract financial valuation commentary.
    """
    
    def __init__(self, delay: float = 1.0, use_selenium: bool = False):
        self.session = requests.Session()
        self.delay = delay
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.logger = self._setup_logger()
        
        if self.use_selenium:
            self.driver = None
            self._setup_selenium()
        else:
            # Rotate user agents to appear more human-like
            self.user_agents = [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
            
            # Set initial headers
            self._update_headers()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver with Chrome options optimized for Link11 bypass."""
        try:
            chrome_options = Options()
            
            # Don't run headless - Link11 can detect headless browsers
            # chrome_options.add_argument("--headless")
            
            # Essential options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Realistic user agent
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Enable JavaScript (needed for Link11 challenges)
            # chrome_options.add_argument("--disable-javascript")
            
            # Additional options to appear more human-like
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Enable all features that make the browser look real
            chrome_options.add_argument("--enable-javascript")
            chrome_options.add_argument("--enable-images")
            chrome_options.add_argument("--enable-plugins")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Selenium: {e}")
            self.use_selenium = False
            self.driver = None
    
    def _update_headers(self):
        """Update session headers with a random user agent and realistic headers."""
        import random
        
        user_agent = random.choice(self.user_agents)
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"'
        }
        
        self.session.headers.update(headers)
    
    def _setup_logger(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common non-content elements
        text = re.sub(r'[^\w\s\.\,\-\+\$\%\(\)\:]+', '', text)
        
        # Remove empty lines and excessive spacing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content using minimal, robust selectors."""
        content_parts = []
        
        # Key selectors for financial content (under 25 total)
        selectors = [
            # Main content containers
            'div[data-s="topPage"]',
            'div[data-s="description"]', 
            'div[data-s="kpis"]',
            'div[data-s="earning"]',
            'div[data-s="financialOverview"]',
            'div[data-s="technicalAnalysis"]',
            'div[data-s="riskOverview"]',
            'div[data-s="peersComparison"]',
            'div[data-s="events"]',
            
            # Business overview and revenue sections
            '.company-description',
            '.revenue-model',
            '.business-overview',
            
            # Key metrics and ratios
            '.financial-highlights',
            '.key-ratios',
            '.performance-metrics',
            
            # Analyst content
            '.analyst-ratings',
            '.price-targets',
            '.consensus-ratings',
            
            # Earnings and financial data
            '.earnings-summary',
            '.financial-data',
            '.quarterly-results',
            
            # Company events and news
            '.corporate-events',
            '.company-news'
        ]
        
        # Debug: Check if we're getting any content at all
        total_elements_found = 0
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                total_elements_found += len(elements)
                
                for element in elements:
                    # Remove script, style, and chart elements
                    for unwanted in element.find_all(['script', 'style', 'svg', 'canvas', 'img']):
                        unwanted.decompose()
                    
                    # Remove elements with chart-related classes
                    for chart_elem in element.find_all(class_=re.compile(r'chart|graph|visualization|plot')):
                        chart_elem.decompose()
                    
                    # Extract clean text
                    text = element.get_text(separator='\n', strip=True)
                    cleaned_text = self.clean_text(text)
                    
                    if cleaned_text and len(cleaned_text) > 10:  # Minimum content threshold
                        content_parts.append(cleaned_text)
                        
            except Exception as e:
                self.logger.warning(f"Error processing selector {selector}: {e}")
                continue
        
        # If no content found with selectors, try broader approach
        if not content_parts:
            self.logger.warning(f"No content found with specific selectors. Total elements found: {total_elements_found}")
            
            # Try to find any text content in the main body
            main_content = soup.find('main') or soup.find('body')
            if main_content:
                # Remove all script, style, and interactive elements
                for unwanted in main_content.find_all(['script', 'style', 'svg', 'canvas', 'img', 'nav', 'header', 'footer']):
                    unwanted.decompose()
                
                # Get all text content
                all_text = main_content.get_text(separator='\n', strip=True)
                cleaned_all_text = self.clean_text(all_text)
                
                if cleaned_all_text and len(cleaned_all_text) > 50:  # Higher threshold for broad content
                    content_parts.append(cleaned_all_text)
                    self.logger.info("Used broad content extraction as fallback")
            
            # If still no content, try looking for any div with data-s attributes
            if not content_parts:
                data_s_elements = soup.find_all('div', attrs={'data-s': True})
                self.logger.info(f"Found {len(data_s_elements)} elements with data-s attributes")
                
                for element in data_s_elements:
                    data_s_value = element.get('data-s', '')
                    self.logger.info(f"Found data-s element: {data_s_value}")
                    
                    # Remove unwanted elements
                    for unwanted in element.find_all(['script', 'style', 'svg', 'canvas', 'img']):
                        unwanted.decompose()
                    
                    # Extract text
                    text = element.get_text(separator='\n', strip=True)
                    cleaned_text = self.clean_text(text)
                    
                    if cleaned_text and len(cleaned_text) > 10:
                        content_parts.append(f"[{data_s_value}] {cleaned_text}")
                        self.logger.info(f"Extracted content from {data_s_value}")
        
        return '\n\n'.join(content_parts)
    
    def scrape_tipranks_page(self, url: str) -> Dict[str, any]:
        """
        Scrape TipRanks page and extract financial commentary.
        
        Args:
            url: TipRanks stock analysis URL
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        try:
            self.logger.info(f"Scraping: {url}")
            
            # Add delay to be respectful
            time.sleep(self.delay)
            
            if self.use_selenium:
                return self._scrape_with_selenium(url)
            else:
                return self._scrape_with_requests(url)
                
        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'status': 'error'
            }
    
    def _scrape_with_selenium(self, url: str) -> Dict[str, any]:
        """Scrape using Selenium WebDriver to bypass Link11 protection."""
        try:
            self.logger.info("Using Selenium WebDriver to bypass Link11")
            
            # Navigate to the page
            self.driver.get(url)
            
            # Wait for page to load and handle Link11 challenges
            max_wait = 60  # Increased wait time for Link11 challenges
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    # Check if we're on a Link11 challenge page
                    page_source = self.driver.page_source
                    if "Link11" in page_source or "security threat" in page_source.lower():
                        self.logger.info("Detected Link11 challenge, waiting...")
                        time.sleep(10)  # Wait for challenge to complete
                        self.driver.refresh()  # Refresh the page
                        time.sleep(5)
                        continue
                    
                    # Check if we have actual content
                    if "tipranks" in self.driver.title.lower() and len(page_source) > 10000:
                        break
                    
                    # Wait a bit more
                    time.sleep(5)
                    
                except Exception as e:
                    self.logger.warning(f"Error during page load: {e}")
                    time.sleep(5)
            
            # Final check for Link11 blocking
            final_source = self.driver.page_source
            if "Link11" in final_source or "security threat" in final_source.lower():
                self.logger.error("Still blocked by Link11 after waiting")
                return {
                    'url': url,
                    'error': 'Blocked by Link11 Web Application Security',
                    'status': 'error'
                }
            
            # Wait for page to fully load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(10)
            
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract stock symbol from URL
            stock_symbol = url.split('/')[-2].upper() if '/' in url else 'UNKNOWN'
            
            # Extract main content
            main_content = self.extract_main_content(soup)
            
            # Extract basic metadata
            title = self.driver.title
            title_text = title if title else "No title found"
            
            return {
                'url': url,
                'stock_symbol': stock_symbol,
                'title': title_text,
                'description': '',  # Selenium doesn't easily get meta description
                'content': main_content,
                'content_length': len(main_content),
                'timestamp': time.time(),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Selenium scraping error for {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'status': 'error'
            }
    
    def _scrape_with_requests(self, url: str) -> Dict[str, any]:
        """Scrape using requests library (original method)."""
        try:
            # Try with different approaches if the first fails
            response = None
            for attempt in range(5):  # Increased attempts
                try:
                    # Update headers for each attempt to appear more human-like
                    self._update_headers()
                    
                    if attempt == 0:
                        # First attempt with session
                        response = self.session.get(url, timeout=30)
                    elif attempt == 1:
                        # Second attempt with new session
                        new_session = requests.Session()
                        new_session.headers.update(self.session.headers)
                        response = new_session.get(url, timeout=30)
                    elif attempt == 2:
                        # Third attempt with different user agent
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.9'
                        }
                        response = requests.get(url, headers=headers, timeout=30)
                    elif attempt == 3:
                        # Fourth attempt with Firefox user agent
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.9'
                        }
                        response = requests.get(url, headers=headers, timeout=30)
                    else:
                        # Fifth attempt with minimal headers
                        response = requests.get(url, timeout=30)
                    
                    response.raise_for_status()
                    break
                    
                except requests.RequestException as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 4:  # Last attempt
                        raise e
                    # Exponential backoff
                    wait_time = self.delay * (2 ** attempt)
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract stock symbol from URL
            stock_symbol = url.split('/')[-2].upper() if '/' in url else 'UNKNOWN'
            
            # Debug: Check if we got a valid page
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "No title found"
            
            # Check if we got a valid TipRanks page or if we're being blocked
            if "tipranks" not in title_text.lower() and "stock" not in title_text.lower():
                self.logger.warning(f"Page title doesn't look like TipRanks: {title_text}")
            
            # Extract main content
            main_content = self.extract_main_content(soup)
            
            # Debug: Save raw HTML for inspection if content is empty
            if not main_content:
                import os
                debug_dir = "model/DAPT/data/debug"
                os.makedirs(debug_dir, exist_ok=True)
                debug_file = os.path.join(debug_dir, f"{stock_symbol.lower()}_debug.html")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
                self.logger.info(f"Saved debug HTML to: {debug_file}")
            
            # Extract page description if available
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""
            
            return {
                'url': url,
                'stock_symbol': stock_symbol,
                'title': title_text,
                'description': description,
                'content': main_content,
                'content_length': len(main_content),
                'timestamp': time.time(),
                'status': 'success'
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Request error for {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'status': 'error'
            }
        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'status': 'error'
            }
    
    def save_to_file(self, data: Dict[str, any], filename: str = None) -> str:
        """Save extracted data to a text file."""
        import os
        
        # Create directory if it doesn't exist
        output_dir = "model/DAPT/data/raw-text"
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            stock_symbol = data.get('stock_symbol', 'unknown').lower()
            filename = f"{stock_symbol}_analysis.txt"
        
        # Ensure filename is in the correct directory
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data.get('content', ''))
            
            self.logger.info(f"Data saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving file {filepath}: {e}")
            return None

def test_single_ticker(ticker: str):
    """Test scraping a single ticker with detailed debugging."""
    cleaner = TipRanksCleaner(delay=3.0)
    
    print(f"Testing single ticker: {ticker}")
    print("="*50)
    
    url = f"https://www.tipranks.com/stocks/{ticker.lower()}/stock-analysis"
    result = cleaner.scrape_tipranks_page(url)
    
    print(f"Status: {result['status']}")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Content length: {result.get('content_length', 0)}")
    
    if result['status'] == 'success' and result.get('content_length', 0) == 0:
        print("⚠ Got success but empty content - this suggests the page structure may have changed")
        print("This could be due to:")
        print("1. TipRanks blocking automated requests")
        print("2. Page structure changed")
        print("3. Content is loaded dynamically with JavaScript")
        print("4. The ticker doesn't exist on TipRanks")
    
    return result

def compare_successful_vs_failed():
    """Compare a successful scrape vs a failed one to understand the difference."""
    cleaner = TipRanksCleaner(delay=3.0)
    
    # Test a ticker that worked (AAPL)
    print("Testing successful ticker (AAPL):")
    print("="*50)
    aapl_result = cleaner.scrape_tipranks_page("https://www.tipranks.com/stocks/aapl/stock-analysis")
    
    print(f"AAPL Status: {aapl_result['status']}")
    print(f"AAPL Content Length: {aapl_result.get('content_length', 0)}")
    print(f"AAPL Title: {aapl_result.get('title', 'N/A')}")
    
    print("\n" + "="*50)
    
    # Test a ticker that failed (ORCL)
    print("Testing failed ticker (ORCL):")
    print("="*50)
    orcl_result = cleaner.scrape_tipranks_page("https://www.tipranks.com/stocks/orcl/stock-analysis")
    
    print(f"ORCL Status: {orcl_result['status']}")
    print(f"ORCL Content Length: {orcl_result.get('content_length', 0)}")
    print(f"ORCL Title: {orcl_result.get('title', 'N/A')}")
    
    # Compare the differences
    print("\n" + "="*50)
    print("COMPARISON:")
    print(f"AAPL Title contains 'tipranks': {'tipranks' in aapl_result.get('title', '').lower()}")
    print(f"ORCL Title contains 'tipranks': {'tipranks' in orcl_result.get('title', '').lower()}")
    print(f"AAPL Title contains 'stock': {'stock' in aapl_result.get('title', '').lower()}")
    print(f"ORCL Title contains 'stock': {'stock' in orcl_result.get('title', '').lower()}")
    
    return aapl_result, orcl_result

def test_link11_bypass():
    """Test different approaches to bypass Link11 protection."""
    print("Testing Link11 bypass strategies...")
    print("="*60)
    
    # Test with Selenium (non-headless)
    print("1. Testing with Selenium (non-headless)...")
    cleaner_selenium = TipRanksCleaner(delay=5.0, use_selenium=True)
    result = cleaner_selenium.scrape_tipranks_page("https://www.tipranks.com/stocks/aapl/stock-analysis")
    
    print(f"Status: {result['status']}")
    print(f"Content Length: {result.get('content_length', 0)}")
    print(f"Title: {result.get('title', 'N/A')}")
    
    if cleaner_selenium.driver:
        cleaner_selenium.driver.quit()
    
    return result

def main():
    """Scrape multiple tickers from TipRanks."""
    # Try Selenium first, fall back to requests if not available
    use_selenium = SELENIUM_AVAILABLE
    if use_selenium:
        print("Using Selenium WebDriver to bypass bot protection...")
    else:
        print("Selenium not available, using requests library...")
    
    cleaner = TipRanksCleaner(delay=5.0, use_selenium=use_selenium)  # Increased delay to be more respectful
    
    # List of tickers to scrape
    tickers = [
        'ADBE', 'TFC',
        'CARR', 'AFL', 'ROP', 'GLW', 'KMI', 'TEL', 'EFX', 'AWK',
        'NRG', 'UAL', 'V', 'T', 'STT', 'KHC', 'J', 'STLD', 'BIIB',
        'ON', 'TSN', 'HST', 'BRK.B', 'ORCL'
    ]
    
    successful_scrapes = 0
    failed_scrapes = 0
    
    print(f"Starting to scrape {len(tickers)} tickers...")
    print("="*60)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Scraping {ticker}...")
        
        # Construct URL for the ticker
        url = f"https://www.tipranks.com/stocks/{ticker.lower()}/stock-analysis"
        
        # Scrape the page
        result = cleaner.scrape_tipranks_page(url)
        
        if result['status'] == 'success':
            content_length = result['content_length']
            if content_length > 0:
                print(f"✓ Successfully scraped {result['stock_symbol']}")
                print(f"  Content length: {content_length} characters")
                
                # Save to file
                filepath = cleaner.save_to_file(result)
                if filepath:
                    print(f"  Data saved to: {filepath}")
                
                successful_scrapes += 1
            else:
                print(f"⚠ Scraped {result['stock_symbol']} but got empty content")
                print(f"  This might indicate the page structure changed or content is blocked")
                failed_scrapes += 1
                
        else:
            print(f"✗ Failed to scrape {ticker}: {result.get('error', 'Unknown error')}")
            failed_scrapes += 1
        
        # Add extra delay between tickers to be respectful
        if i < len(tickers):  # Don't delay after the last ticker
            extra_delay = 10  # Additional delay between tickers
            print(f"  Waiting {cleaner.delay + extra_delay} seconds before next request...")
            time.sleep(extra_delay)
    
    print("\n" + "="*60)
    print(f"Scraping completed!")
    print(f"Successful: {successful_scrapes}")
    print(f"Failed: {failed_scrapes}")
    print(f"Total: {len(tickers)}")
    
    if failed_scrapes > 0:
        print("\nTips for failed scrapes:")
        print("1. Install Selenium: pip install selenium")
        print("2. Install ChromeDriver for Selenium")
        print("3. Increase the delay between requests")
        print("4. Try running again later")
        print("5. Check if specific tickers are available on TipRanks")
    
    # Clean up Selenium driver if used
    if cleaner.use_selenium and cleaner.driver:
        cleaner.driver.quit()

if __name__ == "__main__":
    # Uncomment the line below to test a single ticker with detailed debugging
    # test_single_ticker("ORCL")
    
    # Uncomment the line below to compare successful vs failed scrapes
    # compare_successful_vs_failed()
    
    # Uncomment the line below to test Link11 bypass strategies
    # test_link11_bypass()
    
    main()
