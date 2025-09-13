import streamlit as st
import json
from pathlib import Path
from pdf_processor import process_multiple_pdfs, answer_query_intelligently
from arxiv_fetcher import fetch_arxiv_paper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_uploaded_files(uploaded_files):
    """Save uploaded PDF files."""
    Path("pdfs").mkdir(exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        pdf_path = f"pdfs/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(pdf_path)
    return saved_paths

def load_documents():
    """Load documents safely."""
    try:
        with open("extracted_data.json", "r", encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    except:
        return []

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")

st.title("📄 Document Q&A - Simple & Working")
st.markdown("**Upload a PDF and get real answers - no equations, no mixed documents!**")

# Sidebar
with st.sidebar:
    st.header("📊 Status")
    
    documents = st.session_state.get("documents", [])
    if documents:
        current_doc = documents[-1]  
        st.success(f"✅ Document loaded")
        st.info(f"📄 {len(current_doc.get('text', []))} pages")
        st.info(f"📊 {len(current_doc.get('tables', []))} tables")
        
        doc_name = Path(current_doc.get('path', '')).name
        st.caption(f"Current: {doc_name[:30]}...")
    else:
        st.info("📭 No document loaded")

# Main area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📤 Upload PDF")
    uploaded_files = st.file_uploader("Choose PDF", type="pdf", accept_multiple_files=False)
    
    if uploaded_files:
        if "documents" in st.session_state:
            del st.session_state.documents
        
        saved_paths = save_uploaded_files([uploaded_files])  # Convert single file to list
        
        with st.spinner("Processing PDF..."):
            try:
                process_multiple_pdfs("pdfs")
                all_docs = load_documents()
                
                if all_docs:
                    st.session_state.documents = [all_docs[-1]]
                    current_doc = st.session_state.documents[0]
                    
                    st.success(f"✅ Processed: {Path(current_doc['path']).name}")
                    
                    pages = len(current_doc.get('text', []))
                    tables = len(current_doc.get('tables', []))
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Pages", pages)
                    with col_b:
                        st.metric("Tables", tables)
                else:
                    st.error("❌ No content extracted")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with col2:
    st.subheader("📚 ArXiv")
    arxiv_query = st.text_input("Search:", placeholder="transformers")
    
    if st.button("Fetch", use_container_width=True):
        if arxiv_query:
            # Clear previous documents
            if "documents" in st.session_state:
                del st.session_state.documents
            
            with st.spinner("Fetching..."):
                try:
                    pdf_path = fetch_arxiv_paper(arxiv_query, max_results=1)
                    if pdf_path:
                        st.success(f"✅ Downloaded")
                        
                        process_multiple_pdfs("pdfs")
                        all_docs = load_documents()
                        
                        if all_docs:
                            # Only keep the most recent
                            st.session_state.documents = [all_docs[-1]]
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Q&A Section
documents = st.session_state.get("documents", [])

if documents:
    st.markdown("---")
    st.subheader("❓ Ask Questions")
    
    # Quick buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📋 Summary", use_container_width=True):
            st.session_state.query = "What is the summary of this paper?"
    with col2:
        if st.button("🎯 Accuracy", use_container_width=True):
            st.session_state.query = "What accuracy is reported?"
    with col3:
        if st.button("📝 Conclusion", use_container_width=True):
            st.session_state.query = "What are the conclusions?"
    with col4:
        if st.button("🔬 Method", use_container_width=True):
            st.session_state.query = "What methodology was used?"
    
    # Query input
    query = st.text_input(
        "Your question:",
        value=st.session_state.get("query", ""),
        placeholder="What methodology was used?"
    )
    
    if "query" in st.session_state:
        del st.session_state.query
    
    if query:
        current_doc = documents[0]  
        
        st.info(f"📄 Analyzing: {Path(current_doc['path']).name}")
        
        with st.spinner("Finding answer..."):
            try:
                # Use ONLY the current document
                answer = answer_query_intelligently(current_doc, query)
                
                st.success("💡 **Answer:**")
                st.markdown(f"""
                <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0066cc;">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Tables
    st.markdown("---")
    st.subheader("📊 Tables")
    
    if st.button("Show Tables", use_container_width=True):
        tables = documents[0].get("tables", [])
        
        if tables:
            st.success(f"✅ Found {len(tables)} table(s)")
            
            for i, table in enumerate(tables):
                st.markdown(f"**Table {i+1}** ({len(table)} rows × {len(table[0])} columns)")
                
                try:
                    import pandas as pd
                    if len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.table(table)
                except:
                    st.table(table)
                
                if i < len(tables) - 1:
                    st.markdown("---")
        else:
            st.info("No tables found - this means we avoided fake tables!")

else:
    st.markdown("---")
    st.info("👆 **Upload a PDF above to start asking questions!**")
    
    st.markdown("""
        
    ### 💡 **Try asking:**
    - "What methodology was used?"
    - "What accuracy is reported?" 
    - "What are the conclusions?"
    - "What datasets were used?"
    """)

st.markdown("---")
st.markdown("*Simple, working document Q&A* 🎯")