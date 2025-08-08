#!/usr/bin/env python3

import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://valueinvestorsclub.com")
        print("🟢 Page loaded. Interact manually.")
        
        await asyncio.sleep(90)

if __name__ == "__main__":
    asyncio.run(main())