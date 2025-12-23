import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from dateutil import parser as dateparser  # for flexible date parsing
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

BASE_URL = "https://web-scraping.dev"

# helper date function
def parse_date(date_text: str):
    """
    Try to parse any date string into a datetime object.
    Returns None if parsing fails.
    """
    if not date_text:
        return None
    try:
        return dateparser.parse(date_text)
    except Exception:
        return None
    
# scrape products 

def scrape_products(max_pages: int = 10):
    products = []

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/products?page={page}"
        print(f"Scraping products page {page}: {url}")
        resp = requests.get(url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Each product is inside: <div class="col-8 description">
        product_blocks = soup.select("div.col-8.description")

        # If no products found, stop pagination
        if not product_blocks:
            print("No products found on this page, stopping pagination.")
            break

        for block in product_blocks:
            # Product name + URL
            a_tag = block.find("a")
            name = a_tag.get_text(strip=True) if a_tag else None
            product_url = a_tag["href"] if (a_tag and a_tag.has_attr("href")) else None

            # Make URL absolute if needed
            if product_url and product_url.startswith("/"):
                product_url = BASE_URL + product_url

            # Short description
            desc_tag = block.find("div", class_="short-description")
            short_description = desc_tag.get_text(strip=True) if desc_tag else None

            products.append({
                "name": name,
                "url": product_url,
                "short_description": short_description,
                "page": page,
                "scraped_at": datetime.utcnow().isoformat()
            })

    return products

# scrape testimonials 

def scrape_testimonials_infinite_scroll():
    """
    Scrape all testimonials from the /testimonials page which uses infinite scroll.
    Uses Selenium to scroll through the page and load all testimonials.
    """
    url = f"{BASE_URL}/testimonials"
    print(f"Scraping testimonials with infinite scroll: {url}")
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome()
    testimonials = []
    
    try:
        driver.get(url)
        time.sleep(2)  # Wait for initial page load
        
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait for new content to load
            time.sleep(2)
            
            # Calculate new height
            new_height = driver.execute_script("return document.body.scrollHeight")
            
            # If height hasn't changed, we've reached the end
            if new_height == last_height:
                print("Reached end of testimonials, no more content to load.")
                break
            
            last_height = new_height
        
        # Parse the fully loaded page
        soup = BeautifulSoup(driver.page_source, "html.parser")
        blocks = soup.select("div.testimonial")
        
        print(f"Found {len(blocks)} testimonials")
        
        for block in blocks:
            # Extract testimonial text
            text_tag = block.find("p", class_="text")
            text = text_tag.get_text(strip=True) if text_tag else None
            
            # Extract rating by counting stars (SVGs inside span.rating)
            rating_block = block.find("span", class_="rating")
            stars = len(rating_block.find_all("svg")) if rating_block else 0
            
            # Extract username (from <identicon-svg username="...">)
            identicon = block.find("identicon-svg")
            username = identicon.get("username") if identicon else None
            
            testimonials.append({
                "username": username,
                "rating": stars,
                "text": text,
                "scraped_at": datetime.utcnow().isoformat()
            })
    
    finally:
        driver.quit()
    
    return testimonials

# scrape reviews 

def scrape_reviews_infinite_scroll(max_reviews: int = None):
    """
    Scrape reviews from the /reviews page which uses a "Load More" button.
    Uses Selenium to scroll, click the button, and load all reviews.
    """
    url = f"{BASE_URL}/reviews"
    print(f"Scraping reviews with Load More button: {url}")
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome()
    reviews = []
    
    try:
        driver.get(url)
        time.sleep(2)  # Wait for initial page load
        
        while True:
            # Scroll to bottom first
            print("Scrolling to bottom of page...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for page to settle
            
            # Try to find and click the "Load More" button
            try:
                load_more_button = driver.find_element(By.ID, "page-load-more")
                
                # Check if button is displayed
                if load_more_button.is_displayed():
                    print("Found Load More button, clicking...")
                    load_more_button.click()
                    time.sleep(3)  # Wait for new reviews to appear
                    
                    # Additional wait to ensure all content is loaded
                    time.sleep(2)
                else:
                    print("Load More button is no longer visible, all reviews loaded.")
                    break
            except Exception as e:
                print(f"Load More button not found or clickable: {e}")
                print("All reviews have been loaded.")
                break
            
            # Stop if we've reached max_reviews limit (check early to avoid unnecessary clicks)
            if max_reviews:
                current_reviews = driver.find_elements(By.CSS_SELECTOR, "div.review[data-testid='review']")
                if len(current_reviews) >= max_reviews:
                    print(f"Reached maximum reviews limit of {max_reviews}")
                    break
        
        # Parse the fully loaded page
        soup = BeautifulSoup(driver.page_source, "html.parser")
        review_blocks = soup.select("div.review[data-testid='review']")
        
        print(f"Found {len(review_blocks)} reviews total")
        
        for block in review_blocks:
            # Review ID from data-review-id
            review_id = block.get("data-review-id")
            
            # Date
            date_el = block.find("span", {"data-testid": "review-date"})
            date_text = date_el.get_text(strip=True) if date_el else None
            parsed_date = parse_date(date_text) if date_text else None
            
            # Stars: count SVGs inside span[data-testid="review-stars"]
            stars_span = block.find("span", {"data-testid": "review-stars"})
            star_count = len(stars_span.find_all("svg")) if stars_span else 0
            
            # Text: p[data-testid="review-text"]
            text_el = block.find("p", {"data-testid": "review-text"})
            text = text_el.get_text(strip=True) if text_el else None
            
            reviews.append({
                "review_id": review_id,
                "date_raw": date_text,
                "date_parsed": parsed_date.isoformat() if parsed_date else None,
                "stars": star_count,
                "text": text,
                "scraped_at": datetime.utcnow().isoformat()
            })
            
            # Stop if we've reached max_reviews limit
            if max_reviews and len(reviews) >= max_reviews:
                break
    
    finally:
        driver.quit()
    
    return reviews if not max_reviews else reviews[:max_reviews]

def main():
    #products = scrape_products(max_pages=6)
    # pd.DataFrame(products).to_csv("products.csv", index=False)

    #testimonials = scrape_testimonials_infinite_scroll()
    #pd.DataFrame(testimonials).to_csv("testimonials.csv", index=False)

    reviews = scrape_reviews_infinite_scroll()
    pd.DataFrame(reviews).to_csv("reviews.csv", index=False)

if __name__ == "__main__":
    main()
