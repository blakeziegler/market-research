import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
import os
from urllib.parse import urlparse, unquote
import re

async def main():


    output_dir = "/Users/blakeziegler/tech/Projects/market-research/model/DAPT/data/raw-text"
    os.makedirs(output_dir, exist_ok=True)

    pdf_scraping_cfg = PDFContentScrapingStrategy(
        extract_images=False,
        batch_size=5
    )

    pdf_crawler_cfg = PDFCrawlerStrategy()

    # Per Crawl4AI docs, provide the PDF scraping strategy via `scraping_strategy`
    run_cfg = CrawlerRunConfig(
        scraping_strategy=pdf_scraping_cfg,
    )

    async with AsyncWebCrawler(crawler_strategy=pdf_crawler_cfg) as crawler:
        pdf_url = "file:///Users/blakeziegler/tech/Projects/market-research/model/DAPT/data/book/val_models.pdf"
        print(f"Crawling {pdf_url}")
        result = await crawler.arun(url=pdf_url, config=run_cfg)
        if result.success:
            # Extract markdown text per docs
            md_text = ""
            if getattr(result, "markdown", None) is not None and hasattr(result.markdown, "raw_markdown"):
                md_text = result.markdown.raw_markdown or ""
            else:
                md_text = str(getattr(result, "markdown", "") or "")

            # Post-process markdown: remove specific noisy phrase and '* * *' separators
            md_text = re.sub(r"Applied e quity An A lysis A nd p ortfolio M A n A ge M ent_", "", md_text, flags=re.IGNORECASE)
            # Remove Morgan Stanley copyright footer, entire line (handles trailing page numbers)
            md_text = re.sub(r"© 2024 Morgan Stanley. All rights reserved. 3927037 Exp. 10/31/2025", "", md_text, flags=re.IGNORECASE)
            # Remove lines made of '* * *' separators
            md_text = re.sub(r"(?m)^\s*(?:\*\s*){3,}\s*$", "", md_text)
            # Collapse excessive blank lines left by removals
            md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()

            # Determine a filename based on the PDF name
            parsed = urlparse(pdf_url)
            pdf_path = unquote(parsed.path) if parsed.scheme else pdf_url
            base_name = os.path.splitext(os.path.basename(pdf_path))[0] or "pdf_output"
            output_path = os.path.join(output_dir, f"sector_valuation_models.txt")

            # Save parsed text to the defined output directory
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"Saved parsed PDF text to {output_path}")
        else:
            err = getattr(result, "error_message", None) or getattr(result, "error", "Unknown error")
            print(f"Failed to crawl {pdf_url}: {err}")

if __name__ == "__main__":
    asyncio.run(main())