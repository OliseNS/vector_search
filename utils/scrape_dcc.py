"""
Simple web scraper for DCC Dialysis website.
Clean, straightforward approach to extract text content from web pages.
"""

import os
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse
import json


class DCCSiteCrawler:
    """
    Crawler for dccdialysis.com that discovers and saves all internal pages.
    Each page is saved as both .txt (main text) and .json (url, title, text).
    """
    def __init__(self, base_url="https://dccdialysis.com", delay=1.0, out_dir=None):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(self.base_url).netloc
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.visited = set()
        self.delay = delay
        if out_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            out_dir = os.path.join(base_dir, "data", "raw")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_page(self, url):
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_main_text(self, soup):
        # Try to find main content area
        for selector in ['main', '.main-content', '.content', 'article', 'body']:
            main = soup.select_one(selector)
            if main:
                break
        else:
            main = soup
        # Remove script/style
        for tag in main(["script", "style"]):
            tag.decompose()
        return self.clean_text(main.get_text())

    def extract_title(self, soup):
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        return ""

    def slugify(self, url):
        path = urlparse(url).path.strip('/')
        if not path:
            return "home"
        slug = re.sub(r'[^a-zA-Z0-9_-]', '-', path)
        return slug.lower()[:80]

    def is_internal(self, url):
        parsed = urlparse(urljoin(self.base_url, url))
        return parsed.netloc == self.domain

    def crawl(self):
        queue = [self.base_url]
        self.visited = set()
        print(f"Starting crawl at {self.base_url}")
        while queue:
            url = queue.pop(0)
            if url in self.visited:
                continue
            print(f"Crawling: {url}")
            html = self.get_page(url)
            if not html:
                continue
            soup = BeautifulSoup(html, 'html.parser')
            title = self.extract_title(soup)
            text = self.extract_main_text(soup)
            slug = self.slugify(url)
            # Save .txt
            txt_path = os.path.join(self.out_dir, f"{slug}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            # Save .json
            json_path = os.path.join(self.out_dir, f"{slug}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"url": url, "title": title, "text": text}, f, ensure_ascii=False, indent=2)
            self.visited.add(url)
            # Find new links
            for a in soup.find_all('a', href=True):
                link = urljoin(self.base_url, a['href'])
                if self.is_internal(link) and link not in self.visited and link not in queue:
                    if '#' in link:
                        link = link.split('#')[0]
                    if link not in self.visited and link not in queue:
                        queue.append(link)
            time.sleep(self.delay)
        print(f"Crawling complete. {len(self.visited)} pages saved to {self.out_dir}")


def main():
    crawler = DCCSiteCrawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
