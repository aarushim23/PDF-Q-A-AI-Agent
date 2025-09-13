**Document Q&A AI Agent Prototype**

This repository contains a Document Q&A AI Agent prototype developed as part of the assessment for Task 1: Building an Enterprise-Ready AI Agent. The solution leverages Python and advanced NLP techniques to process PDF documents, extract structured content, and provide intelligent query responses. It includes an ArXiv integration for fetching papers, aligning with enterprise-grade requirements for scalability and usability.

**Overview**
The Document Q&A AI Agent is designed to ingest multiple PDF documents, extract meaningful content (text, tables, and sections), and enable users to query the data via a Streamlit-based interface. It uses the Flan-T5-small LLM for natural language understanding and includes a bonus ArXiv API integration for dynamic paper lookup.

**Features**
Multi-PDF Ingestion: Processes multiple PDF files, extracting text, tables, and metadata.
Content Extraction: Preserves titles, abstracts, sections, and tables with high accuracy, handling equations and figures where possible.
NLP-Powered Queries:
Direct lookup (e.g., "What is the conclusion of Paper X?")
Summarization (e.g., "Summarize the methodology of Paper C.")
Metric extraction (e.g., "What are the accuracy and F1-score reported in Paper D?")


ArXiv Integration: Fetches papers based on user descriptions using the ArXiv API (bonus feature).
Enterprise-Ready: Optimized for context handling, error resilience, and clear documentation.

**Installation**
Prerequisites

Python 3.8 or higher
Conda environment (recommended)

**Setup Instructions**

Clone the repository:git clone <your-github-repo-link>
cd document-qa-ai-agent


Create and activate a Conda environment:conda create -n ai-agent python=3.9
conda activate ai-agent


Install required libraries:pip install pdfplumber transformers torch streamlit arxiv requests


(Optional) For OCR support or advanced LLM:conda install -c conda-forge tesseract
pip install pytesseract mistralai



**Usage**

Run the application:streamlit run app.py


Upload one or more PDF files via the interface.
Use the query input or quick buttons (Summary, Accuracy, Conclusion, Method) to ask questions.
Fetch ArXiv papers by entering a description in the sidebar and clicking "Fetch."
View extracted tables or query responses in the main window.

**File Structure**

app.py: Streamlit frontend with upload, query, and ArXiv fetch capabilities. Includes error handling and a clean UI.
pdf_processor.py: Manages PDF ingestion, text extraction, and table detection with strict validation to preserve structure.
llm_processor.py: Implements the Flan-T5-small LLM for query processing, with fallback text analysis and metric extraction.
arxiv_fetcher.py: Handles ArXiv API calls, paper search, and PDF downloads with robust error recovery.
extracted_data.json: Stores processed document content.

**Technical Details**

LLM Integration: Uses Flan-T5-small via the transformers library for NLP tasks, optimized for context-aware responses.
Content Extraction: pdfplumber extracts text and tables, with custom cleaning to handle equations and figures. Tables are validated to avoid misclassification.
ArXiv API: Leverages the arxiv library for real-time paper fetching, with fallback download methods.
Security: Input sanitization and error logging are implemented to meet basic enterprise standards.

**Limitations**

Table Extraction: Limited to text-based tables; image-based tables require OCR (planned enhancement).
LLM Scalability: Flan-T5-small may underperform with complex queries; upgrading to Mistral-7B is recommended (see Enhancements).
Equation Handling: Basic preservation; advanced rendering (e.g., LaTeX) is not yet supported.

**Enhancements**

OCR Support: Integrate pytesseract in pdf_processor.py for image-based table extraction.
Advanced LLM: Switch to Mistral-7B in llm_processor.py for better accuracy (requires GPU).
LaTeX Support: Add LaTeX rendering for equations using external libraries.
Multi-User Support: Implement session management for enterprise deployment.
