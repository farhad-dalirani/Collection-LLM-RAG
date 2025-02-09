import json
import requests
from bs4 import BeautifulSoup
import re

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            print(f"Skipping {url}: 404 Not Found")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        
        # Extract the main content
        paragraphs = soup.find_all("p")
        text_content = "\n\n".join(p.get_text(strip=False) for p in paragraphs)
        
        # Normalize spaces to ensure no word concatenation happens
        text_content = re.sub(r'\s+', ' ', text_content).strip()  # Replace multiple spaces with a single space
        
        return text_content
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_articles(json_file, output_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    scraped_data = []
    for article in data["data"]:
        name = article.get("Name", "")
        link = article.get("Link", "")
        print(f"Scraping: {name}")
        
        content = extract_text_from_url(link)
        if content:
            article['Content'] = content
            scraped_data.append(article)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=4, ensure_ascii=False)
    
    print(f"Scraping completed. Data saved to {output_file}")


if __name__ == '__main__':
    # Example usage:
    scrape_articles("LLMCONFRAG/create-knowledge-base/input-sources/test.json", "LLMCONFRAG/create-knowledge-base/output-processed-sources/test.json")
