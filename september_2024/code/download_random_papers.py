import arxiv
import random
import os
import time
from datetime import datetime, timedelta
import re

# Function to search for papers on a specific date within the given categories
def search_arxiv_papers(query, max_results=1000):
    """
    Searches for Arxiv papers using the specified query.

    Args:
        query (str): The query string for Arxiv search (includes categories and date range).
        max_results (int): Maximum number of results to retrieve.

    Returns:
        list: A list of Arxiv search results.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    return list(search.results())

# Function to download the full text PDF of a paper
def download_paper(paper, save_dir="arxiv_papers", date="", category=""):
    """
    Downloads the full text PDF of the given Arxiv paper.

    Args:
        paper (arxiv.Result): The Arxiv paper result object.
        save_dir (str): Directory where the PDF should be saved.
        date (str): The date of the paper submission.
        category (str): The category of the paper.

    Returns:
        str: The file path of the downloaded PDF.
    """
    paper_id = paper.entry_id.split('/')[-1]
    paper_id = re.sub(r'v\d+', 'v1', paper_id)
    sanitized_category = category.replace(".", "_")
    file_path = f"{save_dir}/{date}___{sanitized_category}___{paper_id}.pdf"
    paper.download_pdf(dirpath=save_dir, filename=f"{date}___{sanitized_category}___{paper_id}.pdf")
    return file_path

def save_sample(papers, num_papers, categories):
    selected = []
    while len(selected) < num_papers:
        paper = random.choice(papers)
        if paper not in selected and paper.primary_category in categories:
            selected.append(paper)
    return selected

# Function to process and download papers for a specific date
def process_date(date, categories, num_papers=10, save_dir="arxiv_papers", retries=100, retry_delay=15):
    """
    Searches and downloads random papers for a specific date.

    Args:
        date (str): The date in YYYYMMDD format for which to download papers.
        categories (str): The Arxiv categories to search in.
        num_papers (int): Number of random papers to download.
        save_dir (str): Directory where the PDFs should be saved.
        retries (int): Number of retry attempts in case of failure.
        retry_delay (int): Delay in seconds before retrying in case of failure.

    Returns:
        None
    """
    attempt = 0
    while attempt <= retries:
        try:
            # Build the search query for the specified date
            category_query = " OR ".join([f"cat:{cat}" for cat in categories.split()])
            query = f"({category_query}) AND submittedDate:[{date}0000 TO {date}2359]"

            # Search for papers on the specified date
            papers = search_arxiv_papers(query)

            # Check if there are enough papers to download
            if len(papers) < num_papers:
                print(f"Not enough papers found for date {date}. Found {len(papers)} papers.")
                return

            # Randomly select the specified number of papers
            random_papers = save_sample(papers, num_papers, categories.split())

            # Download each selected paper
            for paper in random_papers:
                # Extract the primary category from the paper's categories
                primary_category = paper.primary_category
                print(f"Downloading paper {paper.title} from {date} in category {primary_category}")
                download_paper(paper, save_dir, date=date, category=primary_category)
                time.sleep(3)  # Small delay to be courteous to the server

            # If successful, break out of the retry loop
            break

        except Exception as e:
            attempt += 1
            if attempt <= retries:
                print(f"Error processing date {date}: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt} of {retries})")
                time.sleep(retry_delay)
            else:
                print(f"Failed to process date {date} after {retries} attempts. Error: {e}")
                return

# Main script to download papers over a date range, ignoring weekends
def download_papers_over_range(start_date, end_date, categories, num_papers=10, save_dir="arxiv_papers"):
    """
    Downloads a specified number of random papers each day over a date range, ignoring weekends.

    Args:
        start_date (datetime): The start date of the range.
        end_date (datetime): The end date of the range.
        categories (str): The Arxiv categories to search in.
        num_papers (int): Number of random papers to download per day.
        save_dir (str): Directory where the PDFs should be saved.

    Returns:
        None
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    current_date = start_date
    date_range = []

    # Build a list of dates within the range, skipping weekends
    while current_date < end_date:
        # Check if the current date is a weekday (Monday=0, Sunday=6)
        if current_date.weekday() < 5:  # 0-4 are weekdays
            date_range.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    # Sequentially process each date
    for date in date_range:
        process_date(date, categories, num_papers, save_dir)

if __name__ == "__main__":
    # Parameters
    start_date = datetime(2024, 9, 27)  # Start of the date range
    end_date = datetime(2024, 9, 30)  # End of the date range
    categories = "cs.CL cs.CV cs.AI cs.LG"  # Arxiv categories to search in
    num_papers = 10  # Number of papers to download per day

    # Run the script
    download_papers_over_range(start_date, end_date, categories, num_papers)