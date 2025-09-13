Document Q&A AI Agent
Overview
A Streamlit-based AI agent for querying PDF documents using Flan-T5-small and integrating with the Arxiv API. The agent supports document ingestion, natural language queries, summarization, metrics extraction, and Arxiv paper fetching.
Setup

Clone the repository:git clone <repository-url>
cd doc-qa-agent


Set up Anaconda environment:conda create -n ai-agent python=3.10
conda activate ai-agent
conda install -c conda-forge pdfplumber streamlit pandas transformers torch
pip install arxiv


Run the app:streamlit run app.py



Usage

Upload PDFs: Use the Streamlit interface to upload PDF files.
Query Documents: Ask questions like "What is the conclusion of Paper X?", "Summarize the methodology", or "Extract accuracy".
Fetch Arxiv Papers: Enter a description (e.g., "transformer models 2023") to download and process papers.
View Tables: Display extracted tables from PDFs.

Design Choices

Flan-T5-small: Chosen for lightweight, free LLM processing suitable for CPU environments.
pdfplumber: Used for robust text and table extraction from PDFs.
Streamlit: Provides an intuitive web interface for user interaction.
Arxiv API: Integrated for bonus functionality, enabling dynamic paper fetching.
JSON Storage: Stores extracted PDF content for efficient querying.

Challenges

Managed Flan-T5's 512-token limit by chunking context.
Handled multi-column PDFs with pdfplumber’s robust parsing.
Ensured error handling for Arxiv API and PDF processing.

Future Improvements

Add semantic search using embeddings for better query matching.
Support image/equation extraction with vision models.
Optimize for larger PDFs with database storage.

Video Demo
[Link to YouTube/Google Drive video] (Add after recording)
Testing
Tested with Arxiv papers (e.g., single-column, multi-column, with tables). Edge cases like malformed PDFs and failed Arxiv queries are handled gracefully.