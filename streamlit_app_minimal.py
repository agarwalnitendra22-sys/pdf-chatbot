import streamlit as st
import re
import numpy as np
from typing import List, Tuple

# Try importing with better error handling
missing_packages = []

try:
    import PyPDF2
except ImportError:
    missing_packages.append("PyPDF2")
    PyPDF2 = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    missing_packages.append("scikit-learn")
    sklearn_available = False

try:
    import tiktoken
except ImportError:
    missing_packages.append("tiktoken")
    tiktoken = None

# Set page config
st.set_page_config(
    page_title="PDF Q&A Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #4dabf7);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .search-result {
        background: #f0f7ff;
        border: 1px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .relevance-score {
        background: #e3f2fd;
        color: #1f77b4;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .status-success {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    .info-box {
        background: #e3f2fd;
        border: 1px solid #1f77b4;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: #1f77b4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file."""
    if PyPDF2 is None:
        return ""
    
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def chunk_text_simple(text: str, chunk_size: int = 1500) -> List[str]:
    """Simple text chunking by character count with sentence boundaries."""
    if not text.strip():
        return []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would make chunk too long, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    return [chunk for chunk in chunks if len(chunk) > 100]

def preprocess_text(text: str) -> str:
    """Preprocess text for matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

class DocumentRetriever:
    """Document retrieval system."""
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.use_tfidf = sklearn_available
        
        if self.use_tfidf:
            try:
                processed_chunks = [preprocess_text(chunk) for chunk in chunks]
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                self.chunk_vectors = self.vectorizer.fit_transform(processed_chunks)
            except Exception as e:
                self.use_tfidf = False
    
    def search_document(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant chunks."""
        if self.use_tfidf:
            return self._tfidf_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)
    
    def _tfidf_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """TF-IDF based search."""
        try:
            processed_query = preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append((self.chunks[idx], similarities[idx]))
            
            return results
        except:
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Simple keyword-based search."""
        query_words = set(preprocess_text(query).split())
        
        chunk_scores = []
        for chunk in self.chunks:
            chunk_words = set(preprocess_text(chunk).split())
            if query_words and chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                score = overlap / len(query_words) if query_words else 0
                chunk_scores.append((chunk, score))
            else:
                chunk_scores.append((chunk, 0.0))
        
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:top_k]

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract key terms from text."""
    # Remove common words and clean text
    words = preprocess_text(text).split()
    word_freq = {}
    
    # Count word frequencies, excluding very short words
    for word in words:
        if len(word) > 3:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]

def main():
    # Check for missing packages
    if missing_packages:
        st.markdown("""
        <div class="main-header">
            <h1>📄 PDF Q&A Tool</h1>
            <p>Missing Required Dependencies</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>Missing packages:</strong> {', '.join(missing_packages)}<br><br>
            Please create a requirements.txt file with:<br>
            <pre>streamlit
PyPDF2
scikit-learn
tiktoken
numpy</pre>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize session state
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Free PDF Q&A Tool</h1>
        <p>Upload a PDF and search through its content - No API key required!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        🆓 <strong>Completely Free!</strong> This tool searches your PDF content without requiring any paid API keys.
        Simply upload your PDF and search for specific information.
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.subheader("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF only if it's a new file
        if "current_pdf" not in st.session_state or st.session_state.current_pdf != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text.strip():
                    st.error("Could not extract text from PDF.")
                    return
                
                chunks = chunk_text_simple(pdf_text, 1500)
                
                if not chunks:
                    st.error("No valid text chunks created.")
                    return
                
                retriever = DocumentRetriever(chunks)
                
                # Extract key topics
                keywords = extract_keywords(pdf_text, 15)
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.keywords = keywords
            st.session_state.search_results = []
            
            method = "TF-IDF + Cosine Similarity" if sklearn_available else "Keyword Matching"
            st.markdown(f"""
            <div class="status-success">
                ✅ PDF processed successfully!<br>
                📊 Created {len(chunks)} searchable sections<br>
                🔍 Using {method} for search<br>
                📝 Document length: {len(pdf_text):,} characters
            </div>
            """, unsafe_allow_html=True)
            
            # Show key topics
            if keywords:
                st.markdown("**🏷️ Key topics found in document:**")
                keyword_display = " • ".join(keywords[:10])
                st.markdown(f"*{keyword_display}*")
        
        st.markdown("---")
        
        # Search interface
        st.subheader("🔍 Search Your Document")
        
        # Search form
        with st.form(key="search_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                search_query = st.text_input(
                    "Search for information in your document:",
                    placeholder="e.g., vision, strategy, main goals, key findings...",
                    key="search_input"
                )
            
            with col2:
                search_button = st.form_submit_button("🔍 Search")
                clear_button = st.form_submit_button("🗑️ Clear")
        
        # Process search
        if search_button and search_query.strip():
            with st.spinner("🔍 Searching document..."):
                results = st.session_state.retriever.search_document(search_query, 5)
                st.session_state.search_results = results
                st.session_state.last_query = search_query
        
        # Clear results
        if clear_button:
            st.session_state.search_results = []
            st.success("🗑️ Search results cleared!")
        
        # Display search results
        if st.session_state.search_results:
            st.subheader(f"📋 Search Results for: '{st.session_state.get('last_query', '')}'")
            
            for i, (chunk, score) in enumerate(st.session_state.search_results):
                if score > 0:  # Only show relevant results
                    st.markdown(f"""
                    <div class="search-result">
                        <div class="relevance-score">Relevance: {score:.1%}</div>
                        <strong>Section {i+1}:</strong><br>
                        {chunk[:500]}{'...' if len(chunk) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show full text option
                    if len(chunk) > 500:
                        with st.expander(f"📖 Read full Section {i+1}"):
                            st.write(chunk)
            
            # No relevant results found
            if all(score == 0 for _, score in st.session_state.search_results):
                st.markdown("""
                <div class="warning-box">
                    🤷‍♂️ No highly relevant sections found. Try different keywords or check the key topics above.
                </div>
                """, unsafe_allow_html=True)
        
        # Usage tips
        with st.expander("💡 Search Tips"):
            st.markdown("""
            **How to search effectively:**
            - Use specific keywords from your document
            - Try different variations of terms
            - Look at the key topics shown above for guidance
            - Use 2-4 words for best results
            
            **Example searches:**
            - "company vision" → finds vision statements
            - "financial results" → finds financial data
            - "recommendations" → finds suggested actions
            - "methodology" → finds research methods
            """)
        
        # Document statistics
        if st.session_state.chunks:
            st.markdown(f"""
            <div class="info-box">
                📊 <strong>Document Stats:</strong> 
                {len(st.session_state.chunks)} sections • 
                {len(st.session_state.keywords)} key topics identified •
                Search method: {'TF-IDF Similarity' if sklearn_available else 'Keyword Matching'}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Upload prompt
        st.markdown("""
        ### 📤 Upload Your PDF Document
        
        **This is a FREE tool** that helps you search and find information in your PDF documents without requiring any API keys or subscriptions.
        
        **What you can do:**
        - 🔍 **Search** for specific topics, keywords, or concepts
        - 📊 **View relevance scores** to see how well results match your query
        - 📖 **Read full sections** that contain your searched terms
        - 🏷️ **Discover key topics** automatically extracted from your document
        
        **Perfect for:**
        - Research papers and reports
        - Business documents and manuals
        - Legal documents and contracts
        - Academic papers and thesis documents
        - Technical documentation
        
        **No limits, no costs, no API keys needed!**
        """)

if __name__ == "__main__":
    main()
