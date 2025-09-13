#!/usr/bin/env python3
"""
Debug script to test the document Q&A system components - Fixed version
"""

import sys
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_processing():
    """Test PDF processing functionality"""
    print("🔍 Testing PDF Processing...")
    
    try:
        from pdf_processor import process_multiple_pdfs, extract_pdf_content
        
        # Check if pdfs directory exists
        pdf_dir = Path("pdfs")
        if not pdf_dir.exists():
            print("❌ 'pdfs' directory not found. Creating it...")
            pdf_dir.mkdir()
            print("📁 Please add some PDF files to the 'pdfs' directory and run this test again.")
            return False
        
        # Check for PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found in 'pdfs' directory.")
            print("📁 Please add some PDF files and try again.")
            return False
        
        print(f"✅ Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        
        # Test processing
        print("📄 Processing PDFs...")
        results = process_multiple_pdfs("pdfs")
        
        if results:
            print(f"✅ Successfully processed {len(results)} PDFs")
            
            # Show stats
            for result in results:
                print(f"   📋 {Path(result['path']).name}:")
                print(f"      - Pages: {len(result['text'])}")
                print(f"      - Tables: {len(result['tables'])}")
                
                # Show sample table data
                if result['tables']:
                    sample_table = result['tables'][0]
                    print(f"      - Sample table: {len(sample_table)} rows x {len(sample_table[0]) if sample_table else 0} cols")
                    # Show first few cells
                    if sample_table and sample_table[0]:
                        sample_cells = [str(cell)[:20] + "..." if len(str(cell)) > 20 else str(cell) for cell in sample_table[0][:3]]
                        print(f"      - Sample data: {sample_cells}")
        else:
            print("❌ No results from PDF processing")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install pdfplumber camelot-py[cv] pandas")
        return False
    except Exception as e:
        print(f"❌ Error during PDF processing: {e}")
        logger.exception("PDF processing error details:")
        return False

def test_llm_setup():
    """Test LLM setup and functionality"""
    print("\n🤖 Testing LLM Setup...")
    
    try:
        from llm_processor import setup_llm, query_llm
        
        print("📥 Loading language model...")
        llm = setup_llm()
        
        if not llm:
            print("❌ Failed to load LLM")
            return False
        
        print(f"✅ LLM loaded successfully: {llm.get('model_name', 'Unknown')}")
        print(f"🔧 Device: {llm.get('device', 'Unknown')}")
        
        if llm.get('fallback', False):
            print("⚠️  Using fallback text processing mode")
        
        # Test query
        test_context = """
        This research paper presents a new deep learning model for image classification. 
        The proposed model achieved 94.5% accuracy on the CIFAR-10 dataset and 87.2% accuracy on ImageNet. 
        The methodology combines convolutional neural networks with attention mechanisms.
        The main conclusion is that attention mechanisms significantly improve classification performance.
        """
        
        test_queries = [
            "What is the accuracy?",
            "What methodology was used?",
            "What is the conclusion?"
        ]
        
        print("🔍 Testing sample queries...")
        all_responses_good = True
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            try:
                response = query_llm(llm, test_context, query)
                print(f"💬 Response: {response}")
                
                if not response or len(response.strip()) < 10:
                    print("⚠️  Warning: Query returned very short response")
                    all_responses_good = False
                elif "error" in response.lower():
                    print("⚠️  Warning: Error detected in response")
                    all_responses_good = False
                    
            except Exception as e:
                print(f"❌ Error with query '{query}': {e}")
                all_responses_good = False
        
        return all_responses_good
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure transformers and torch are installed:")
        print("   pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Error during LLM testing: {e}")
        logger.exception("LLM testing error details:")
        return False

def test_arxiv_fetcher():
    """Test ArXiv fetching functionality"""
    print("\n📚 Testing ArXiv Fetcher...")
    
    try:
        from arxiv_fetcher import fetch_arxiv_paper, search_arxiv_papers
        
        print("🔍 Testing ArXiv search functionality...")
        
        # Test search first (doesn't download)
        test_query = "attention mechanism"
        papers = search_arxiv_papers(test_query, max_results=2)
        
        if papers:
            print(f"✅ Search successful! Found {len(papers)} papers:")
            for i, paper in enumerate(papers, 1):
                title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
                print(f"   {i}. {title}")
                print(f"      Authors: {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}")
        else:
            print("❌ Search returned no results")
            return False
        
        print("\n📥 Testing paper download (this may take a moment)...")
        
        # Test download with first search result
        pdf_path = fetch_arxiv_paper(test_query, max_results=1)
        
        if pdf_path and Path(pdf_path).exists():
            file_size = Path(pdf_path).stat().st_size
            print(f"✅ Successfully downloaded paper: {Path(pdf_path).name}")
            print(f"📄 File size: {file_size / 1024:.1f} KB")
            
            if file_size < 1000:  # Less than 1KB is suspicious
                print("⚠️  Warning: Downloaded file seems too small")
                return False
                
            return True
        else:
            print("❌ Failed to download paper or file doesn't exist")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure arxiv package is installed:")
        print("   pip install arxiv")
        return False
    except Exception as e:
        print(f"❌ Error during ArXiv testing: {e}")
        logger.exception("ArXiv testing error details:")
        return False

def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n🔄 Testing Full Pipeline...")
    
    try:
        # Check if we have processed data
        if not Path("extracted_data.json").exists():
            print("❌ No extracted_data.json found. Run PDF processing first.")
            return False
        
        # Load processed data with proper encoding
        try:
            with open("extracted_data.json", "r", encoding='utf-8') as f:
                documents = json.load(f)
        except UnicodeDecodeError:
            try:
                with open("extracted_data.json", "r", encoding='latin-1') as f:
                    documents = json.load(f)
                print("⚠️  Loaded with latin-1 encoding")
            except Exception:
                with open("extracted_data.json", "r", encoding='utf-8', errors='ignore') as f:
                    documents = json.load(f)
                print("⚠️  Loaded with error ignoring")
        
        if not documents:
            print("❌ No documents in extracted_data.json")
            return False
        
        print(f"✅ Loaded {len(documents)} processed documents")
        
        # Show document info
        for i, doc in enumerate(documents):
            print(f"   📄 Document {i+1}: {Path(doc['path']).name}")
            print(f"      - Text pages: {len(doc.get('text', []))}")
            print(f"      - Tables: {len(doc.get('tables', []))}")
        
        # Test with real document data
        from llm_processor import setup_llm, query_llm
        
        llm = setup_llm()
        if not llm:
            print("❌ Failed to load LLM for pipeline test")
            return False
        
        # Use first document
        doc = documents[0]
        text_pages = doc.get("text", [])
        if not text_pages:
            print("❌ No text content in document")
            return False
        
        # Combine first few pages
        combined_text = " ".join(text_pages[:3])  # Use first 3 pages
        if len(combined_text) < 100:
            print("⚠️  Very little text content found")
        
        print(f"📊 Working with {len(combined_text)} characters of text")
        
        print("🔍 Testing pipeline with real document...")
        
        # Test different query types
        queries = [
            ("General", "What is this paper about?"),
            ("Results", "What are the main results?"),
            ("Methods", "What methodology was used?")
        ]
        
        all_queries_successful = True
        
        for query_type, query in queries:
            print(f"\n❓ {query_type}: {query}")
            try:
                # Limit context to avoid token limits
                context = combined_text[:1500] if len(combined_text) > 1500 else combined_text
                response = query_llm(llm, context, query)
                
                if response and len(response.strip()) > 10:
                    # Show truncated response
                    display_response = response[:150] + "..." if len(response) > 150 else response
                    print(f"💬 {display_response}")
                else:
                    print("⚠️  Got empty or very short response")
                    all_queries_successful = False
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
                all_queries_successful = False
        
        # Test table information
        tables = doc.get("tables", [])
        if tables:
            print(f"\n📊 Table Analysis:")
            print(f"   Found {len(tables)} tables")
            
            for i, table in enumerate(tables[:2]):  # Show first 2 tables
                print(f"   📋 Table {i+1}: {len(table)} rows × {len(table[0]) if table else 0} columns")
                if table and len(table) > 0:
                    # Show first row as headers
                    headers = [str(cell)[:15] + "..." if len(str(cell)) > 15 else str(cell) for cell in table[0][:4]]
                    print(f"      Headers: {headers}")
        else:
            print("📊 No tables found in document")
        
        return all_queries_successful
        
    except Exception as e:
        print(f"❌ Error during pipeline testing: {e}")
        logger.exception("Pipeline testing error details:")
        return False

def test_dependencies():
    """Test if all required packages are installed"""
    print("📦 Testing Dependencies...")
    
    required_packages = [
        ("pdfplumber", "PDF text extraction"),
        ("transformers", "Language models"),
        ("torch", "PyTorch for transformers"),
        ("arxiv", "ArXiv API access"),
        ("requests", "HTTP requests"),
        ("pandas", "Data manipulation")
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (MISSING)")
            missing_packages.append(package)
    
    # Test optional packages
    optional_packages = [
        ("camelot", "Advanced table extraction")
    ]
    
    print("\n📦 Optional Dependencies:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ⚠️  {package} - {description} (optional, but recommended)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main debug function"""
    print("🔧 Document Q&A System Debug Tool - Fixed Version")
    print("=" * 60)
    
    # Test dependencies first
    print("Step 1: Checking dependencies...")
    deps_ok = test_dependencies()
    if not deps_ok:
        print("\n❌ Please install missing dependencies before proceeding.")
        return
    
    print("\n" + "=" * 60)
    
    # Test components
    tests = [
        ("PDF Processing", test_pdf_processing),
        ("LLM Setup", test_llm_setup),
        ("ArXiv Fetcher", test_arxiv_fetcher),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Step {len(results) + 2}: {test_name}")
        print("=" * 60)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            logger.exception(f"Exception in {test_name}:")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system should work correctly.")
        print("💡 You can now run: streamlit run app.py")
    elif passed >= total * 0.75:  # 75% pass rate
        print("\n⚠️  Most tests passed. The system should work with some limitations.")
        print("💡 You can try running: streamlit run app.py")
    else:
        print("\n❌ Multiple tests failed. Please fix the issues before using the system.")
        
    print("\n🔧 Common fixes:")
    print("   - Install missing packages: pip install -r requirements.txt")
    print("   - Add PDF files to the 'pdfs' directory for testing")
    print("   - Check your internet connection for ArXiv fetching")
    print("   - Make sure you have enough disk space for model downloads")

if __name__ == "__main__":
    main()