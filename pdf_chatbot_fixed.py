import streamlit as st
import anthropic
import re
import datetime
import os
import numpy as np
from typing import List, Tuple
import tiktoken

# Try importing optional dependencies with fallbacks
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.error("PyPDF2 not found. Please install it: pip install PyPDF2")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_SUPPORT = True
except ImportError:
    SKLEARN_SUPPORT = False
    st.warning("scikit-learn not found. Using fallback retrieval method.")

# Set page config with blue theme
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for blue & white theme
st.markdown("""
<style>
    /* Main background */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #4dabf7);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Upload section */
    .upload-section {
        background: #f8f9ff;
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border: 1px solid #e3f2fd;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* User message */
    .user-message {
        background: #1f77b4;
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        word-wrap: break-word;
    }
    
    /* Bot message */
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
    
    /* Status indicators */
    .status-success {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border: 1px solid #1f77b4;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1f77b4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background: #1565c0;
    }
    
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling for file uploader */
    .uploadedFile {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation (4 chars = 1 token)
        return len(text) // 4

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    if not PDF_SUPPORT:
        st.error("PDF support not available. Please install PyPDF2.")
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

def chunk_text_by_tokens(text: str, min_tokens: int = 500, max_tokens: int = 800, overlap_tokens: int = 100) -> List[str]:
    """Split text into chunks based on token count."""
    if not text.strip():
        return []
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence would exceed max_tokens, finalize current chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            if current_tokens >= min_tokens:
                chunks.append(current_chunk.strip())
                
                # Create overlap for next chunk
                overlap_chunk = ""
                overlap_tokens_count = 0
                j = i - 1
                
                # Add sentences backwards until we reach overlap limit
                while j >= 0 and overlap_tokens_count < overlap_tokens:
                    prev_sentence = sentences[j]
                    prev_tokens = count_tokens(prev_sentence)
                    if overlap_tokens_count + prev_tokens <= overlap_tokens:
                        overlap_chunk = prev_sentence + " " + overlap_chunk
                        overlap_tokens_count += prev_tokens
                    else:
                        break
                    j -= 1
                
                current_chunk = overlap_chunk
                current_tokens = overlap_tokens_count
            else:
                # Current chunk too small, just add sentence
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
                i += 1
        else:
            # Add sentence to current chunk
            current_chunk += " " + sentence
            current_tokens += sentence_tokens
            i += 1
    
    # Add final chunk if it meets minimum requirements
    if current_chunk.strip() and current_tokens >= min_tokens:
        chunks.append(current_chunk.strip())
    
    # Filter out chunks that are too short
    valid_chunks = []
    for chunk in chunks:
        if count_tokens(chunk) >= min_tokens:
            valid_chunks.append(chunk)
    
    return valid_chunks

def preprocess_text_for_tfidf(text: str) -> str:
    """Preprocess text for TF-IDF vectorization."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

class SmartRetriever:
    """Smart retrieval system using TF-IDF and cosine similarity."""
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.processed_chunks = [preprocess_text_for_tfidf(chunk) for chunk in chunks]
        
        if SKLEARN_SUPPORT:
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),  # Include both unigrams and bigrams
                min_df=1,
                max_df=0.95
            )
            
            # Fit and transform chunks
            try:
                self.chunk_vectors = self.vectorizer.fit_transform(self.processed_chunks)
            except ValueError as e:
                st.error(f"Error in TF-IDF vectorization: {e}")
                self.chunk_vectors = None
        else:
            self.vectorizer = None
            self.chunk_vectors = None
    
    def retrieve_top_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k most similar chunks using cosine similarity."""
        if self.chunk_vectors is None or not SKLEARN_SUPPORT:
            # Fallback to simple keyword matching
            return self._fallback_retrieval(query, top_k)
        
        # Preprocess query
        processed_query = preprocess_text_for_tfidf(query)
        
        try:
            # Transform query using fitted vectorizer
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return chunks with similarity scores
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include chunks with some similarity
                    results.append((self.chunks[idx], similarities[idx]))
            
            return results if results else self._fallback_retrieval(query, top_k)
            
        except Exception as e:
            return self._fallback_retrieval(query, top_k)
    
    def _fallback_retrieval(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Fallback keyword-based retrieval."""
        query_words = set(preprocess_text_for_tfidf(query).split())
        
        chunk_scores = []
        for chunk in self.chunks:
            chunk_words = set(preprocess_text_for_tfidf(chunk).split())
            if query_words and chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                score = overlap / len(query_words)
                chunk_scores.append((chunk, score))
            else:
                chunk_scores.append((chunk, 0.0))
        
        # Sort by score and return top-k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:top_k]

def check_question_relevance(question: str, top_chunks: List[Tuple[str, float]], threshold: float = 0.1) -> bool:
    """Check if question is relevant based on top chunk similarities."""
    if not top_chunks:
        return False
    
    # Check if any chunk has similarity above threshold
    max_similarity = max(score for _, score in top_chunks)
    return max_similarity >= threshold

def query_claude_with_smart_retrieval(client, question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """Query Claude API with smart-retrieved chunks."""
    
    if not retrieved_chunks or not check_question_relevance(question, retrieved_chunks):
        return "This question is outside the scope of the manual."
    
    # Extract chunks (ignore similarity scores for context)
    chunks = [chunk for chunk, score in retrieved_chunks]
    
    # Combine relevant chunks
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks):
        context_parts.append(f"Document Section {i+1}:\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    # Enhanced prompt with retrieval information
    prompt = f"""You are a document assistant that answers questions based ONLY on the provided document sections.

CRITICAL INSTRUCTIONS:
1. Only use information from the document sections below
2. If the answer is not in the provided sections, respond: "This question is outside the scope of the manual."
3. Be precise and helpful in your responses
4. Keep responses concise but complete
5. When possible, reference which document section contains the information

Document Sections:
{context}

User Question: {question}

Answer based only on the document sections above:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Updated to latest Claude model
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
        
    except Exception as e:
        return f"Error processing your question: {str(e)}"

def display_chat_messages():
    """Display chat messages with custom styling."""
    if "messages" in st.session_state and st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">💬 Start a conversation by asking a question about your PDF!</div>', unsafe_allow_html=True)

def main():
    # Check dependencies first
    if not PDF_SUPPORT:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Missing Dependencies</strong><br>
            Please install the required packages:<br>
            <code>pip install PyPDF2 scikit-learn tiktoken</code>
        </div>
        """, unsafe_allow_html=True)
        
        # Show installation instructions
        st.markdown("""
        ### 📦 Installation Instructions
        
        Create a `requirements.txt` file with the following content:
        ```
        streamlit
        anthropic
        PyPDF2
        scikit-learn
        tiktoken
        numpy
        ```
        
        Then run:
        ```bash
        pip install -r requirements.txt
        ```
        """)
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
    
    # API Key Section
    with st.container():
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "Enter your Anthropic API Key:", 
            type="password", 
            help="Get your API key from https://console.anthropic.com/",
            placeholder="sk-ant-..."
        )
        
        if not api_key:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>API Key Required</strong><br>
                Please enter your Anthropic API key to continue. You can get one from 
                <a href="https://console.anthropic.com/" target="_blank">console.anthropic.com</a>
            </div>
            """, unsafe_allow_html=True)
            return
    
    # Initialize Claude client
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error initializing Claude client: {str(e)}")
        return
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document to start chatting about its content"
    )
    
    if uploaded_file is not None:
        pdf_filename = uploaded_file.name
        
        # Check if this is a new file
        if "current_pdf" not in st.session_state or st.session_state.current_pdf != pdf_filename:
            with st.spinner("🔄 Processing PDF and setting up retrieval system..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text.strip():
                    st.error("❌ Could not extract text from the PDF. Please try a different file.")
                    return
                
                # Create token-based chunks
                chunks = chunk_text_by_tokens(pdf_text, 500, 800)
                
                if not chunks:
                    st.error("❌ No valid chunks created from the PDF.")
                    return
                
                # Initialize smart retriever
                retriever = SmartRetriever(chunks)
            
            # Store data in session state
            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.session_state.client = client
            st.session_state.pdf_filename = pdf_filename
            st.session_state.current_pdf = pdf_filename
            
            # Clear previous messages when new PDF is uploaded
            st.session_state.messages = []
            
            retrieval_method = "TF-IDF + Cosine Similarity" if SKLEARN_SUPPORT else "Keyword Matching"
            
            st.markdown(f"""
            <div class="status-success">
                ✅ Successfully processed "{pdf_filename}"<br>
                📊 Created {len(chunks)} text chunks<br>
                🔍 Using {retrieval_method} for retrieval
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Chat Section
        st.subheader("💬 Chat with your Document")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            display_chat_messages()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input and controls
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question about your document:",
                placeholder="e.g., What are the main topics discussed?",
                key="chat_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
            reset_chat = st.button("🔄 Reset Chat", help="Clear conversation history")
        
        # Process user input
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                # Smart retrieval
                retrieved_chunks = st.session_state.retriever.retrieve_top_chunks(user_input, 3)
                
                # Query Claude
                response = query_claude_with_smart_retrieval(
                    st.session_state.client, 
                    user_input, 
                    retrieved_chunks
                )
            
            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear input and refresh
            st.rerun()
        
        # Reset chat functionality
        if reset_chat:
            st.session_state.messages = []
            st.success("🗑️ Conversation history cleared!")
            st.rerun()
        
        # Chat statistics
        if st.session_state.messages:
            user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            out_of_scope = len([msg for msg in st.session_state.messages 
                              if msg["role"] == "assistant" and "outside the scope" in msg["content"]])
            st.markdown(f"""
            <div class="info-box">
                📊 <strong>Session Stats:</strong> {user_messages} questions asked, {out_of_scope} out-of-scope responses
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Show upload prompt
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Your PDF Document</h3>
            <p>Click the button above to upload a PDF document and start asking questions about its content.</p>
            <br>
            <p><strong>✨ Features:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>🔤 Smart text chunking (500-800 tokens)</li>
                <li>🔍 TF-IDF + Cosine similarity retrieval</li>
                <li>🤖 Powered by Claude AI</li>
                <li>💬 Clean chat interface</li>
                <li>🛡️ Document-only responses (no hallucination)</li>
                <li>📊 Real-time processing statistics</li>
            </ul>
            <br>
            <p><strong>📝 Example Questions:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>"What are the main topics discussed?"</li>
                <li>"Can you summarize the key points?"</li>
                <li>"What does the document say about [specific topic]?"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()