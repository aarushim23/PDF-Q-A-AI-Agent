import pdfplumber
import json
from pathlib import Path
import re
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text_simple(text: str) -> str:
    """Simple text cleaning that actually works."""
    if not text:
        return ""
    
    # Remove bad characters
    text = str(text)
    bad_chars = ['\x81', '\x8d', '\x8f', '\x90', '\x9d', '\x00']
    for char in bad_chars:
        text = text.replace(char, '')
    
    # Fix broken lines
    text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', text)  # Fix hyphenated words
    text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)  # Join broken sentences
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Fix paragraph breaks
    text = re.sub(r'\s+', ' ', text)  # Fix multiple spaces
    
    return text.strip()

def is_real_table(table_data: List[List[str]]) -> bool:
    """Super strict table detection - only real tables."""
    if not table_data or len(table_data) < 2:
        return False
    
    # Size limits
    if len(table_data) > 30 or len(table_data[0]) > 10:
        return False
    
    # Check for paragraph content in cells
    for row in table_data[:3]:  # Check first 3 rows
        for cell in row[:3]:  # Check first 3 cells
            if not cell:
                continue
            cell_text = str(cell).lower()
            
            # Reject if cell contains paragraph indicators
            paragraph_words = ['this paper', 'we present', 'our approach', 'abstract', 
                             'introduction', 'the proposed', 'however', 'therefore']
            
            if any(word in cell_text for word in paragraph_words):
                return False
            
            # Reject if cell is too long (likely paragraph)
            if len(cell_text) > 80:
                return False
    
    # Must have some numbers to be a real table
    all_text = ' '.join(str(cell) for row in table_data for cell in row if cell)
    numbers = re.findall(r'\b\d+\.?\d*\b', all_text)
    
    if len(numbers) < 2:  # Need at least 2 numbers
        return False
    
    return True

def extract_simple_tables(pdf_path: str) -> List[List[List[str]]]:
    """Extract only obvious, real tables."""
    tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Only try line-based (most reliable)
                settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 2
                }
                
                try:
                    page_tables = page.find_tables(table_settings=settings)
                    
                    for table in page_tables:
                        extracted = table.extract()
                        if extracted and is_real_table(extracted):
                            # Clean the table
                            clean_table = []
                            for row in extracted:
                                clean_row = [clean_text_simple(str(cell)) if cell else "" for cell in row]
                                if any(cell.strip() for cell in clean_row):
                                    clean_table.append(clean_row)
                            
                            if clean_table:
                                tables.append(clean_table)
                                logger.info(f"Found real table on page {page_num}: {len(clean_table)} rows")
                
                except Exception as e:
                    logger.warning(f"Error on page {page_num}: {e}")
    
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    
    return tables

def find_methodology_simple(text: str) -> str:
    """Simple keyword-based methodology extraction."""
    text_lower = text.lower()
    
    # Look for methodology section
    method_section = ""
    
    # Find methodology section
    patterns = [
        r'methodology\s*[:\n]([^\.]*(?:\.[^\.]*){0,5})',
        r'method\s*[:\n]([^\.]*(?:\.[^\.]*){0,3})',
        r'approach\s*[:\n]([^\.]*(?:\.[^\.]*){0,3})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.DOTALL)
        if matches:
            method_section = matches[0]
            break
    
    if not method_section:
        # Look for method sentences
        method_sentences = []
        
        # Find sentences with method keywords
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) < 20 or len(sentence_lower) > 200:
                continue
            
            # Method indicators
            if any(phrase in sentence_lower for phrase in [
                'we use', 'we employ', 'we apply', 'we adopt', 'we implement',
                'our method', 'our approach', 'our algorithm', 'our technique',
                'the algorithm', 'the method', 'the approach'
            ]):
                # Avoid equations and symbols
                if not re.search(r'[𝑛𝔸∈∀∃∈∉⊂⊆∪∩]|[\[\]{}]|\$|\\\w+', sentence):
                    method_sentences.append(sentence.strip())
        
        if method_sentences:
            return "Methodology: " + " ".join(method_sentences[:2])
    else:
        # Clean the method section
        method_section = re.sub(r'[𝑛𝔸∈∀∃∈∉⊂⊆∪∩]|[\[\]{}]|\$|\\\w+', '', method_section)
        if len(method_section.strip()) > 20:
            return "Methodology: " + method_section.strip()
    
    return "Methodology not clearly described in accessible language."

def find_accuracy_simple(text: str) -> str:
    """Simple accuracy extraction."""
    # Direct patterns for accuracy
    patterns = [
        r'accuracy[:\s]*([0-9]+\.?[0-9]*%?)',
        r'([0-9]+\.?[0-9]*%?)\s*accuracy',
        r'f1[:\s]*([0-9]+\.?[0-9]*)',
        r'precision[:\s]*([0-9]+\.?[0-9]*)',
        r'recall[:\s]*([0-9]+\.?[0-9]*)',
        r'achieves[:\s]*([0-9]+\.?[0-9]*%?)',
        r'performance[:\s]*([0-9]+\.?[0-9]*%?)',
    ]
    
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            results.append(match)
    
    if results:
        # Remove duplicates
        unique_results = list(dict.fromkeys(results))
        return "Performance: " + ", ".join(unique_results[:3])
    
    return "No specific performance metrics found."

def find_conclusion_simple(text: str) -> str:
    """Simple conclusion extraction."""
    text_lower = text.lower()
    
    # Look for conclusion section
    conclusion_patterns = [
        r'conclusion[s]?\s*[:\n]([^\.]*(?:\.[^\.]*){0,3})',
        r'in conclusion[^\.]*\.',
        r'we conclude[^\.]*\.',
        r'summary[:\n]([^\.]*(?:\.[^\.]*){0,2})',
    ]
    
    for pattern in conclusion_patterns:
        matches = re.findall(pattern, text_lower, re.DOTALL)
        if matches:
            conclusion = matches[0] if isinstance(matches[0], str) else matches[0][0]
            # Clean and return
            conclusion = re.sub(r'[𝑛𝔸∈∀∃∈∉⊂⊆∪∩]|[\[\]{}]|\$|\\\w+', '', conclusion)
            if len(conclusion.strip()) > 20:
                return "Conclusion: " + conclusion.strip()
    
    # Look for concluding sentences
    sentences = re.split(r'[.!?]+', text)
    conclusion_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if len(sentence_lower) < 20 or len(sentence_lower) > 200:
            continue
        
        if any(phrase in sentence_lower for phrase in [
            'we conclude', 'in conclusion', 'our findings', 'results show',
            'this shows', 'we demonstrate', 'this work shows'
        ]):
            # Avoid equations
            if not re.search(r'[𝑛𝔸∈∀∃∈∉⊂⊆∪∩]|[\[\]{}]|\$|\\\w+', sentence):
                conclusion_sentences.append(sentence.strip())
    
    if conclusion_sentences:
        return "Conclusion: " + " ".join(conclusion_sentences[:2])
    
    return "Conclusion not clearly stated in accessible language."

def answer_query_simple(pdf_content: Dict[str, Any], query: str) -> str:
    """Simple keyword-based answering that works."""
    if not pdf_content or not pdf_content.get('text'):
        return "No content available to answer the query."
    
    # Get the current document's text ONLY
    current_text = " ".join(pdf_content['text'])
    current_text = clean_text_simple(current_text)
    
    if not current_text:
        return "No readable text found in this document."
    
    query_lower = query.lower()
    
    # Route to appropriate handler
    if any(word in query_lower for word in ["method", "approach", "technique", "algorithm"]):
        return find_methodology_simple(current_text)
    
    elif any(word in query_lower for word in ["accuracy", "performance", "result", "score"]):
        return find_accuracy_simple(current_text)
    
    elif any(word in query_lower for word in ["conclusion", "summary", "finding"]):
        return find_conclusion_simple(current_text)
    
    else:
        # General keyword search
        query_words = [word for word in re.findall(r'\b\w{4,}\b', query_lower)
                      if word not in ['what', 'how', 'why', 'when', 'where']]
        
        if not query_words:
            return "Please ask a more specific question."
        
        # Find sentences with query keywords
        sentences = re.split(r'[.!?]+', current_text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if (len(sentence) > 30 and len(sentence) < 300 and
                any(word in sentence_lower for word in query_words)):
                # Skip equations
                if not re.search(r'[𝑛𝔸∈∀∃∈∉⊂⊆∪∩]|[\[\]{}]|\$|\\\w+', sentence):
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return "Answer: " + " ".join(relevant_sentences[:2])
        
        return "Could not find clear information about this topic in the document."

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """Extract content simply and reliably."""
    logger.info(f"Processing: {pdf_path}")
    
    content = {
        "path": str(pdf_path),
        "text": [],
        "tables": [],
        "metadata": {}
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text() or ""
                    cleaned_text = clean_text_simple(page_text)
                    content["text"].append(cleaned_text)
                    logger.info(f"Page {page_num}: {len(cleaned_text)} chars")
                except Exception as e:
                    logger.warning(f"Error on page {page_num}: {e}")
                    content["text"].append("")
            
            # Extract metadata
            try:
                if pdf.metadata:
                    content["metadata"] = {k: str(v) for k, v in pdf.metadata.items() if k and v}
            except:
                pass
        
        # Extract tables
        content["tables"] = extract_simple_tables(pdf_path)
        logger.info(f"Found {len(content['tables'])} real tables")
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    
    return content

def process_multiple_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    """Process PDFs simply."""
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        return []
    
    all_content = []
    for pdf_path in pdf_files:
        try:
            content = extract_pdf_content(str(pdf_path))
            if content and content["text"]:
                all_content.append(content)
                logger.info(f"✅ Processed {pdf_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed {pdf_path.name}: {e}")
    
    # Save results
    try:
        with open("extracted_data.json", "w", encoding='utf-8', errors='replace') as f:
            json.dump(all_content, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(all_content)} documents")
    except Exception as e:
        logger.error(f"Save error: {e}")
    
    return all_content

# Keep the same function name for compatibility
def answer_query_intelligently(pdf_content: Dict[str, Any], query: str) -> str:
    """Wrapper to maintain compatibility."""
    return answer_query_simple(pdf_content, query)

if __name__ == "__main__":
    # Quick test
    results = process_multiple_pdfs("pdfs")
    print(f"Processed {len(results)} PDFs")
    
    if results:
        test_queries = [
            "What methodology was used?",
            "What accuracy was achieved?", 
            "What are the conclusions?"
        ]
        
        for query in test_queries:
            answer = answer_query_simple(results[0], query)
            print(f"\nQ: {query}")
            print(f"A: {answer}")