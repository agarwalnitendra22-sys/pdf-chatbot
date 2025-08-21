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
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except:
            pass
    # Fallback: word count approximation
    return len(text.split())

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

def chunk_text_simple(text: str, chunk_size: int = 1000) -> List[str]:
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
    return [chunk for chunk in chunks if len(chunk) > 200]

def preprocess_text(text: str) -> str:
    """Preprocess text for matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

class SimpleRetriever:
    """Simple retrieval system."""
    
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
                st.warning(f"TF-IDF setup failed: {e}")
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
                results.append((self.chunks[idx], similarities[idx]))
            
            return results
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
        return "I couldn't find any relevant information in the document to answer your question."
    
    # Use a much lower threshold - even small similarity can be useful
    max_score = max(score for _, score in retrieved_chunks)
    
    # Always try to answer if we have chunks, regardless of similarity score
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks[:3]):  # Limit to top 3
        context_parts.append(f"Document Section {i+1}:\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided document sections. 

Please read through the document sections below and answer the user's question. If the information needed to answer the question is not clearly present in any of the sections, you can say so, but try your best to provide a helpful response based on what is available.

Document Sections:
{context}

User Question: {question}

Please provide a helpful answer based on the document sections above:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            temperature=0.2,
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
            Please create a requirements.txt file with:<br>
            <pre>streamlit
anthropic
PyPDF2
scikit-learn
tiktoken
numpy</pre>
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
        st.warning("⚠️ Please enter your Anthropic API key to continue.")
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
        # Process PDF only if it's a new file
        if "current_pdf" not in st.session_state or st.session_state.current_pdf != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text.strip():
                    st.error("Could not extract text from PDF.")
                    return
                
                # Use simpler chunking
                chunks = chunk_text_simple(pdf_text, 2000)  # Larger chunks
                
                if not chunks:
                    st.error("No valid text chunks created.")
                    return
                
                retriever = SimpleRetriever(chunks)
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.session_state.client = client
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.messages = []  # Clear messages for new PDF
            
            method = "TF-IDF + Cosine Similarity" if sklearn_available else "Keyword Matching"
            st.success(f"✅ PDF processed! Created {len(chunks)} chunks using {method}")
        
        st.markdown("---")
        
        # Chat interface
        st.subheader("💬 Chat with your Document")
        
        # Display messages in a container to prevent infinite loops
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
        
        # Input form to prevent auto-rerun
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask about your document:",
                    placeholder="e.g., What is the main topic of this document?",
                    key="user_question"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send 📤")
                reset_button = st.form_submit_button("Reset 🔄")
        
        # Process form submission
        if submit_button and user_input.strip():
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                retrieved_chunks = st.session_state.retriever.retrieve_top_chunks(user_input, 3)
                
                # Debug info (remove in production)
                if retrieved_chunks:
                    max_score = max(score for _, score in retrieved_chunks)
                    st.write(f"Debug: Max similarity score: {max_score:.3f}")
                
                response = query_claude(st.session_state.client, user_input, retrieved_chunks)
            
            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to show new messages
            st.rerun()
        
        # Reset functionality
        if reset_button:
            st.session_state.messages = []
            st.success("🗑️ Chat history cleared!")
            st.rerun()
        
        # Show statistics
        if st.session_state.messages:
            user_msg_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            st.markdown(f"""
            <div class="info-box">
                📊 <strong>Chat Stats:</strong> {user_msg_count} questions asked
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Upload prompt
        st.markdown("""
        ### 📤 Upload Your PDF Document
        
        Upload a PDF file above to start chatting with your document using Claude AI.
        
        **Features:**
        - 🔤 Smart text processing
        - 🔍 Intelligent content retrieval  
        - 🤖 Powered by Claude AI
        - 💬 Interactive chat interface
        
        **Example Questions:**
        - "What is this document about?"
        - "What are the main points?"
        - "Summarize the key findings"
        """)

if __name__ == "__main__":
    main()
