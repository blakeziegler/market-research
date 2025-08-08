import os
import re
import random
import dotenv
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai import LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import RateLimiter
from crawl4ai import CrawlerMonitor, DisplayMode
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.async_dispatcher import SemaphoreDispatcher

urls = [
    "https://beyondspx.com/article/anywhere-real-estate-transforming-for-octane-in-a-volatile-market-hous",
    "https://beyondspx.com/article/millrose-properties-unlocking-homebuilder-capital-through-a-differentiated-land-banking-platform-mrp",
    "https://beyondspx.com/article/cbre-reshaping-real-estate-services-for-enduring-growth-cbre",
    "https://beyondspx.com/article/coastal-infill-advantage-terreno-realty-s-strategy-drives-strong-rent-growth-trno",
    "https://beyondspx.com/article/easterly-government-properties-mission-critical-edge-fuels-growth-ambitions-nyse-dea",
    "https://beyondspx.com/article/transcontinental-realty-investors-unpacking-the-value-in-a-diversified-portfolio-nyse-tci",
    "https://beyondspx.com/article/exelon-powering-growth-through-grid-investment-and-data-center-demand-nasdaq-exc",
    "https://beyondspx.com/article/uepco-powering-growth-with-accelerated-investment-and-a-robust-economic-pipeline-uepco",
    "https://beyondspx.com/article/atmos-energy-fueling-growth-through-infrastructure-investment-nyse-ato",
    "https://beyondspx.com/article/criteo-ai-powered-commerce-media-platform-poised-for-growth-amidst-client-transitions-crto",
    "https://beyondspx.com/article/scripps-deleveraging-momentum-meets-strategic-opportunity-nasdaq-ssp",
    "https://beyondspx.com/article/snap-s-ar-ambition-and-ad-platform-evolution-drive-growth-nyse-snap"
]

async def main():
    dotenv.load_dotenv()
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(
            provider="openai/gpt-4o",
            api_token=os.getenv("OPENAI_API_KEY"),
        ),
        extraction_type="block",
        instruction=
        """
Extract only relevant content related to company financials, valuation, analysis, company information, economics, 
monetary policy, politics, social science, math, theoretical ideas,and other relevent information. 
Exclude all non-content elements such as copyright notices, disclaimers, authorship, dates, URLs, headers, footers, sidebars, 
login prompts, images, videos, social media links, charts, webpage settings, content lists, other or related articles, 
and search tools. Omit any website metadata. 
Format any tables clearly for LLM consumption. Include all financial and technical analysis details; discard everything unrelated.
Make sure all urls within the content are removed and no important information is lost. Get rid of all references as well.
Make sure to extract all valutation metrics like P/E, P/B, P/S, EBITA, WACC, Revenue, EPS, etc.
        """,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.15, "max_tokens": 1500},
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=85.0,
        check_interval=1.0,
        max_session_permit=10,
        rate_limiter=RateLimiter(
            base_delay=(9, 12),
            max_delay=70,
            max_retries=3
        )

    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        stream=True,
        markdown_generator=DefaultMarkdownGenerator()
    )

    browser_config = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Stream results so we can save each file as soon as it's ready (per docs)
        saved_count = 0
        total_count = len(urls)
        print(f"Streaming crawl for {total_count} urls...")

        idx = 0
        async for result in await crawler.arun_many(
            urls=urls,
            config=crawl_config,
            dispatcher=dispatcher
        ):
            idx += 1
            if not getattr(result, "success", False):
                err = getattr(result, "error_message", None) or getattr(result, "error", "")
                print(f"[{idx}] Skipped (error): {err}")
                continue

            # Prefer extracted JSON/text; always fallback to markdown
            content_raw = (result.extracted_content or "").strip()
            if not content_raw:
                content_raw = (result.markdown or "").strip()
            if not content_raw:
                print(f"[{idx}] Empty content for {getattr(result, 'url', '')}")
                continue

            def try_parse_json(text: str) -> Any:
                t = text.strip()
                # Strip code fences if present
                if t.startswith("```") and t.endswith("```"):
                    lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
                    t = "\n".join(lines).strip()
                # Try direct JSON
                try:
                    return json.loads(t)
                except Exception:
                    pass
                # Try to locate a JSON array/object within the text
                for opener, closer in [("[", "]"), ("{", "}")]:
                    start = t.find(opener)
                    end = t.rfind(closer)
                    if start != -1 and end != -1 and end > start:
                        snippet = t[start:end + 1]
                        try:
                            return json.loads(snippet)
                        except Exception:
                            continue
                return None

            def flatten(obj: Any) -> List[str]:
                lines: List[str] = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        # Skip noisy fields
                        if key in {"error", "index"}:
                            continue
                        if key == "tags" and isinstance(value, list):
                            tag_str = ", ".join(str(v) for v in value)
                            if tag_str:
                                lines.append(f"Tags: {tag_str}")
                            continue
                        if key == "content" and isinstance(value, list):
                            for v in value:
                                if v is None:
                                    continue
                                text = str(v).strip()
                                if text:
                                    lines.append(text)
                            continue
                        if isinstance(value, (dict, list)):
                            lines.extend(flatten(value))
                        else:
                            val = str(value).strip()
                            if val:
                                lines.append(f"{key}: {val}")
                elif isinstance(obj, list):
                    for item in obj:
                        lines.extend(flatten(item))
                        if lines and lines[-1] != "":
                            lines.append("")
                else:
                    s = str(obj).strip()
                    if s:
                        lines.append(s)
                return lines

            # Remove obvious non-JSON trailers (e.g., token usage reports) before parsing
            content_for_parse = re.split(r"^===\s*Token Usage Summary\s*===", content_raw, flags=re.MULTILINE)[0]
            parsed = try_parse_json(content_for_parse)
            flattened_text = "\n".join(flatten(parsed)) if parsed is not None else content_raw

            # If flattening produced too little content, fall back to markdown
            if len(flattened_text.strip()) < 50:
                fallback_md = (result.markdown or "").strip()
                if fallback_md:
                    flattened_text = fallback_md

            # Regex cleanup on flattened text
            flattened_text = re.sub(r'\"?error\"?\s*[:=]\s*false\s*,?', "", flattened_text, flags=re.IGNORECASE)
            flattened_text = re.sub(r'\"?index\"?\s*[:=]\s*\d+\s*,?', "", flattened_text)
            # Remove any tag lines like "Tags: finance_reports"
            flattened_text = re.sub(r'(?m)^\s*Tags:\s*.*$', "", flattened_text)
            # Remove leftover JSON symbols on empty lines
            flattened_text = re.sub(r"^\s*[\[\]\{\}]\s*$", "", flattened_text, flags=re.MULTILINE)
            # Collapse excessive blank lines
            flattened_text = re.sub(r"\n{3,}", "\n\n", flattened_text).strip()

            # Save to randomized filename
            output_dir = "/Users/blakeziegler/tech/Projects/market-research/model/DAPT/data/raw-text"
            os.makedirs(output_dir, exist_ok=True)
            rand_id = random.randint(1, 1000000)
            output_path = os.path.join(output_dir, f"crawl_{rand_id}.txt")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    # Prefix with source URL for traceability
                    source = getattr(result, "url", "")
                    header = f"Source: {source}\n\n" if source else ""
                    f.write(header + flattened_text)
                    f.flush()
                saved_count += 1
                print(f"[{idx}] Saved {len(flattened_text)} chars to {output_path}")
            except Exception as e:
                print(f"[{idx}] Write failed: {e}")

        # Show total LLM usage once
        llm_strategy.show_usage()
        print(f"Saved {saved_count}/{total_count} files.")

if __name__ == "__main__":
    asyncio.run(main())
