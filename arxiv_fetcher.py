import arxiv
import requests
import os
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

def fetch_arxiv_paper(query: str, max_results: int = 1) -> str:
    """
    Fetch a paper from ArXiv based on a search query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to consider (default: 1)
    
    Returns:
        Path to the downloaded PDF file, or None if failed
    """
    try:
        # Ensure pdfs directory exists
        pdf_dir = Path("pdfs")
        pdf_dir.mkdir(exist_ok=True)
        
        logger.info(f"Searching ArXiv for: {query}")
        print(f"🔍 Searching ArXiv for: '{query}'")
        
        # Create search client
        client = arxiv.Client()
        
        # Create search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # Get results
        results = list(client.results(search))
        
        if not results:
            print("❌ No papers found for the given query")
            return None
        
        # Get the first (most relevant) result
        paper = results[0]
        
        print(f"📄 Found paper: {paper.title}")
        print(f"👥 Authors: {', '.join([author.name for author in paper.authors[:3]])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"📅 Published: {paper.published.strftime('%Y-%m-%d')}")
        
        # Create filename (sanitize title)
        safe_title = "".join(c for c in paper.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:50]  # Limit length
        filename = f"{safe_title}_{paper.entry_id.split('/')[-1]}.pdf"
        filepath = pdf_dir / filename
        
        # Download the PDF
        print(f"📥 Downloading PDF...")
        try:
            paper.download_pdf(dirpath=str(pdf_dir), filename=filename)
            
            # Verify file exists and has content
            if filepath.exists() and filepath.stat().st_size > 1000:  # At least 1KB
                print(f"✅ Successfully downloaded: {filename}")
                print(f"📊 File size: {filepath.stat().st_size / 1024:.1f} KB")
                return str(filepath)
            else:
                print("❌ Downloaded file is empty or corrupted")
                if filepath.exists():
                    filepath.unlink()  # Remove corrupted file
                return None
                
        except Exception as download_error:
            print(f"❌ Download failed: {download_error}")
            
            # Try alternative download method
            try:
                print("🔄 Trying alternative download method...")
                pdf_url = paper.pdf_url
                
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                if filepath.stat().st_size > 1000:
                    print(f"✅ Alternative download successful: {filename}")
                    return str(filepath)
                else:
                    print("❌ Alternative download also failed")
                    filepath.unlink()
                    return None
                    
            except Exception as alt_error:
                print(f"❌ Alternative download failed: {alt_error}")
                return None
    
    except Exception as e:
        logger.error(f"Error fetching ArXiv paper: {e}")
        print(f"❌ Error fetching paper: {e}")
        return None

def search_arxiv_papers(query: str, max_results: int = 5):
    """
    Search ArXiv papers and return metadata without downloading.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        List of paper metadata dictionaries
    """
    try:
        logger.info(f"Searching ArXiv for papers: {query}")
        
        # Create search client
        client = arxiv.Client()
        
        # Create search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # Get results
        results = list(client.results(search))
        
        papers = []
        for paper in results:
            paper_info = {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'published': paper.published.strftime('%Y-%m-%d'),
                'pdf_url': paper.pdf_url,
                'entry_id': paper.entry_id,
                'categories': paper.categories
            }
            papers.append(paper_info)
        
        return papers
    
    except Exception as e:
        logger.error(f"Error searching ArXiv papers: {e}")
        return []

def download_paper_by_id(arxiv_id: str) -> str:
    """
    Download a specific paper by ArXiv ID.
    
    Args:
        arxiv_id: ArXiv paper ID (e.g., "2301.12345")
    
    Returns:
        Path to downloaded PDF file, or None if failed
    """
    try:
        # Ensure pdfs directory exists
        pdf_dir = Path("pdfs")
        pdf_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading ArXiv paper by ID: {arxiv_id}")
        print(f"📥 Downloading paper ID: {arxiv_id}")
        
        # Create search client
        client = arxiv.Client()
        
        # Search by ID
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        
        if not results:
            print(f"❌ Paper with ID {arxiv_id} not found")
            return None
        
        paper = results[0]
        
        # Create filename
        filename = f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
        filepath = pdf_dir / filename
        
        # Download
        paper.download_pdf(dirpath=str(pdf_dir), filename=filename)
        
        if filepath.exists() and filepath.stat().st_size > 1000:
            print(f"✅ Successfully downloaded: {filename}")
            return str(filepath)
        else:
            print("❌ Download failed or file is corrupted")
            return None
    
    except Exception as e:
        logger.error(f"Error downloading paper by ID: {e}")
        print(f"❌ Error: {e}")
        return None

# Test function
def test_arxiv_fetcher():
    """Test the ArXiv fetcher functionality."""
    print("Testing ArXiv fetcher...")
    
    # Test search
    test_query = "transformer attention"
    print(f"Searching for: {test_query}")
    
    papers = search_arxiv_papers(test_query, max_results=3)
    
    if papers:
        print(f"✅ Found {len(papers)} papers")
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. {paper['title'][:60]}...")
            print(f"     Authors: {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}")
            print(f"     Published: {paper['published']}")
    else:
        print("❌ No papers found")
        return False
    
    # Test download
    print(f"\nTesting download of first paper...")
    pdf_path = fetch_arxiv_paper(test_query, max_results=1)
    
    if pdf_path:
        print(f"✅ Download test successful")
        return True
    else:
        print("❌ Download test failed")
        return False

if __name__ == "__main__":
    test_arxiv_fetcher()