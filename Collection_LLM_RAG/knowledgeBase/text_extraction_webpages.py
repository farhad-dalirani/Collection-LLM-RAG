import re
import json
import fitz
import logging
import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    """
    Extracts and cleans text content from a given URL.
    This function sends a GET request to the specified URL, parses the HTML content,
    removes unwanted elements (such as scripts, styles, headers, footers, navigation, and asides),
    and extracts the text from paragraph, preformatted, and code elements. The extracted text
    is then normalized to avoid unwanted formatting issues.
    Args:
        url (str): The URL of the webpage to extract text from.
    Returns:
        str: The cleaned and extracted text content from the webpage, or None if an error occurs
             or if the URL returns a 404 Not Found status.
    """

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            logging.warning(f"Skipping {url}: 404 Not Found")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()

        # Extract all relevant elements in the order they appear
        content = []
        for element in soup.find_all(["p", "pre", "code"]):  
            if element.name == "p":
                content.append(element.get_text(strip=False))
            elif element.name in ["pre", "code"]:
                content.append(f"\n```\n{element.get_text(strip=False)}\n```\n")  # Preserve code block formatting

        # Join extracted content while preserving order
        full_content = "\n\n".join(content)

        # Normalize spaces to avoid unwanted formatting issues
        full_content = re.sub(r'\s+', ' ', full_content).strip()
        
        return full_content
    except requests.RequestException as e:
        logging.info(f"Error fetching {url}: {e}")
        return None

def scrape_articles(json_file, output_file):
    """
    Scrapes article content from URLs provided in a JSON file and saves the results to an output file.

    Args:
        json_file (str): Path to the input JSON file containing article names and URLs.
        output_file (str): Path to the output JSON file where scraped content will be saved.

    The function reads the input JSON file, extracts article names and URLs, scrapes the content from each URL,
    and saves the updated data (including the scraped content) into the output JSON file.

    The expected format of the input JSON:
    {
        "description": "Some description",
        "data": [
            {"Name": "Article 1", "Link": "https://example.com/article1"},
            {"Name": "Article 2", "Link": "https://example.com/article2"},
            ...
        ]
    }

    The output JSON will retain the original structure but include the scraped "Content" for each article.

    Prints messages to indicate scraping progress and completion, and then return the path of output file.
    """
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("The file was not found: {}.".format(json_file))
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format: {}.".format(json_file))
    
    scraped_data = []
    for article in data["data"]:
        name = article.get("Name", "")
        link = article.get("Link", "")
        logging.info(f"Scraping: {name}")
        
        content = extract_text_from_url(link)
        if content:
            article['Content'] = content
            scraped_data.append(article)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"description": data["description"], "data": scraped_data}, f, indent=4, ensure_ascii=False)
    
    logging.info(f">   Scraping completed. Data saved to {output_file}")

    return output_file

def extract_text_from_pdf_url(url) -> str:
        """
        Extracts text content from a PDF file located at a given URL.
        This function sends a GET request to the specified URL, downloads the PDF content,
        and extracts the text using the PyMuPDF library.
        
        Args:
            url (str): The URL of the PDF file to extract text from.
        
        Returns:
            str: The extracted text content from the PDF, or None if an error occurs.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Open the PDF from the response content
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            
            # Extract text from each page
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            
            return text.strip()
        except requests.RequestException as e:
            logging.info(f"Error fetching {url}: {e}")
            return None
        except Exception as e:
            logging.info(f"Error processing PDF from {url}: {e}")
            return None
        
def scrape_pdfs(json_file, output_file):
    """
    Scrapes PDF content from URLs provided in a JSON file and saves the results to an output file.
    Args:
        json_file (str): Path to the input JSON file containing article names and URLs.
        output_file (str): Path to the output JSON file where scraped content will be saved.
    The function reads the input JSON file, extracts article names and URLs, scrapes the content from each URL,
    and saves the updated data (including the scraped content) into the output JSON file.
    The expected format of the input JSON:
    {
        "description": "Some description",
        "data": [
            {"Name": "PDF 1", "Link": "https://example.com/pdf1.pdf"},
            {"Name": "PDF 2", "Link": "https://example.com/pdf2.pdf"},
            ...
        ]
    }
    The output JSON will retain the original structure but include the scraped "Content" for each PDF.
    Prints messages to indicate scraping progress and completion, and then return the path of output file.
    """
            
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("The file was not found: {}.".format(json_file))
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format: {}.".format(json_file))
    
    scraped_data = []
    for article in data["data"]:
        name = article.get("Name", "")
        link = article.get("Link", "")
        logging.info(f"Scraping PDF: {name}")
        
        content = extract_text_from_pdf_url(link)
        if content:
            article['Content'] = content
            scraped_data.append(article)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"description": data["description"], "data": scraped_data}, f, indent=4, ensure_ascii=False)
    
    logging.info(f">   PDF scraping completed. Data saved to {output_file}")

    return output_file