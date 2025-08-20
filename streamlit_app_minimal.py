import streamlit as st
import re
import numpy as np
from typing import List, Tuple

# Try importing with better error handling
missing_packages = []

try:
    import anthropic
except ImportError:
    missing_packages.append("anthropic")
    anthropic = None

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
    page_title="PDF Chatbot",
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
    .user-message {
        background: #1f77b4;
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        word-wrap: break-word;
    }
    .bot-message {
        background: #f0f7ff;
        color: #1f77b4;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 4px solid #1f77b4;
        word-wrap: break-word;
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
</style>
""", unsafe_allow_html=True)

def count_tokens_fallback(text: str) -> int:
    """Fallback token counting method."""
    return len(text.split())

def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except:
            pass
    # Fallback method
    return count_tokens_fallback(text)

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

def chunk_text_by_tokens(text: str, min_tokens: int = 500, max_tokens: int = 800) -> List[str]:
    """Split text into chunks."""
    if not text.strip():
        return []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            if current_tokens >= min_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk.strip() and count_tokens(current_chunk) >= min_tokens:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if count_tokens(chunk) >= min_tokens]

def preprocess_text(text: str) -> str:
    """Preprocess text for matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

class SimpleRetriever:
    """Simple keyword-based retrieval system."""
    
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
            except:
                self.use_tfidf = False
    
    def retrieve_top_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant chunks."""
        if self.use_tfidf:
            return self._tfidf_retrieval(query, top_k)
        else:
            return self._keyword_retrieval(query, top_k)
    
    def _tfidf_retrieval(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """TF-IDF based retrieval."""
        try:
            processed_query = preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append((self.chunks[idx], similarities[idx]))
            
            return results if results else self._keyword_retrieval(query, top_k)
        except:
            return self._keyword_retrieval(query, top_k)
    
    def _keyword_retrieval(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Keyword-based retrieval."""
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

def query_claude(client, question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """Query Claude with retrieved chunks."""
    if not retrieved_chunks:
        return "I couldn't find relevant information to answer your question."
    
    # Check relevance
    max_score = max(score for _, score in retrieved_chunks)
    if max_score < 0.1:
        return "This question appears to be outside the scope of the document."
    
    # Prepare context
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks):
        context_parts.append(f"Section {i+1}:\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer the following question based ONLY on the provided document sections. If the answer is not in the sections, say so.

Document Sections:
{context}

Question: {question}

Answer:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error processing question: {str(e)}"

def main():
    # Check for missing packages
    if missing_packages:
        st.markdown("""
        <div class="main-header">
            <h1>📄 PDF Chatbot</h1>
            <p>Missing Required Dependencies</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>Missing packages:</strong> {', '.join(missing_packages)}<br><br>
            
            <strong>For Streamlit Cloud:</strong><br>
            1. Create a <code>requirements.txt</code> file in your repository root<br>
            2. Add the following lines:<br>
            <pre>streamlit
anthropic
PyPDF2
scikit-learn
tiktoken
numpy</pre>
            3. Commit and push the changes<br>
            4. Redeploy your app<br><br>
            
            <strong>For local development:</strong><br>
            Run: <code>pip install {' '.join(missing_packages)}</code>
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 PDF Chatbot</h1>
        <p>Upload a PDF document and ask questions about its content using Claude AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key
    api_key = st.text_input(
        "Enter your Anthropic API Key:", 
        type="password",
        help="Get your API key from https://console.anthropic.com/"
    )
    
    if not api_key:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Please enter your Anthropic API key to continue.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize client
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Claude client: {str(e)}")
        return
    
    st.markdown("---")
    
    # File upload
    st.subheader("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF
        if "current_pdf" not in st.session_state or st.session_state.current_pdf != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text.strip():
                    st.error("Could not extract text from PDF.")
                    return
                
                chunks = chunk_text_by_tokens(pdf_text)
                
                if not chunks:
                    st.error("No valid text chunks created.")
                    return
                
                retriever = SimpleRetriever(chunks)
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.session_state.client = client
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.messages = []
            
            method = "TF-IDF + Cosine Similarity" if sklearn_available else "Keyword Matching"
            st.markdown(f"""
            <div class="status-success">
                ✅ PDF processed successfully!<br>
                📊 Created {len(chunks)} chunks<br>
                🔍 Using {method}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Chat interface
        st.subheader("💬 Chat")
        
        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
        
        # Input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Ask about your document:", key="chat_input")
        with col2:
            reset = st.button("🔄 Reset")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                retrieved_chunks = st.session_state.retriever.retrieve_top_chunks(user_input, 3)
                response = query_claude(st.session_state.client, user_input, retrieved_chunks)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if reset:
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()