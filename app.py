"""
Streamlit Frontend for Astrophysics RAG Chatbot
Friendly interface for exploring astrophysics research papers
"""

import streamlit as st
from ai import AstrophysicsRAGChatbot
import time

# Page configuration
st.set_page_config(
    page_title="Astrophysics Research Assistant",
    page_icon="telescope",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Global Styles with Gradient Background */
    .stApp {
        background: linear-gradient(180deg, #D9E1F1 0%, #8FB3E2 100%);
        font-family: 'Space Grotesk', sans-serif;
        line-height: 1.6;
    }
    
    /* Main Container */
    .main .block-container {
        padding: 3rem 4rem;
        max-width: 1200px;
    }
    
    /* Modern Header */
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 600;
        text-align: center;
        color: #192338;
        margin-bottom: 0.5rem;
        padding: 1.5rem;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(25, 35, 56, 0.1);
    }
    
    .sub-header {
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
        color: #1E2E4F;
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 3rem;
        letter-spacing: 0.5px;
    }
    
    /* Modern Dark Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1E2E4F;
        border-right: 1px solid #31487A;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF;
    }
    
    section[data-testid="stSidebar"] strong {
        color: #8FB3E2;
    }
    
    section[data-testid="stSidebar"] li {
        color: #D9E1F1;
    }
    
    /* Sidebar Buttons - Modern Card Style */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #31487A;
        color: #FFFFFF;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        text-align: left;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #8FB3E2;
        color: #192338;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(143, 179, 226, 0.3);
    }
    
    /* Sidebar Metrics */
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #8FB3E2;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
        color: #FFFFFF;
    }
    
    /* Sidebar Divider */
    section[data-testid="stSidebar"] hr {
        border-color: #31487A;
        opacity: 0.5;
    }
    
    /* Chat Messages - Modern Cards */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(25, 35, 56, 0.08);
        border: 1px solid rgba(143, 179, 226, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stChatMessage p {
        font-family: 'Space Grotesk', sans-serif;
        color: #192338;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Source Box */
    .source-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 14px;
        border-left: 4px solid #31487A;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(25, 35, 56, 0.08);
        backdrop-filter: blur(10px);
    }
    
    /* Modern Buttons */
    .stButton > button {
        background-color: #31487A;
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        font-size: 1.05rem;
        border-radius: 10px;
        border: none;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background-color: #1E2E4F;
        box-shadow: 0 6px 18px rgba(49, 72, 122, 0.35);
        transform: translateY(-2px);
    }
    
    /* Modern Text Input */
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: #192338;
        border: 2px solid #8FB3E2;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #31487A;
        box-shadow: 0 0 0 4px rgba(49, 72, 122, 0.1);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #8FB3E2;
        opacity: 0.8;
    }
    
    /* Slider Styling */
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: #31487A;
    }
    
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #8FB3E2;
    }
    
    /* Checkbox */
    section[data-testid="stSidebar"] .stCheckbox > label {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF;
        font-size: 1rem;
    }
    
    section[data-testid="stSidebar"] .stCheckbox input[type="checkbox"]:checked {
        background-color: #8FB3E2;
    }
    
    /* Metrics in Main Area */
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        color: #31487A;
        font-weight: 600;
        font-size: 2rem;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'Space Grotesk', sans-serif;
        color: #1E2E4F;
        font-size: 1.05rem;
    }
    
    /* Divider in Main Area */
    .main hr {
        border-color: rgba(49, 72, 122, 0.2);
        opacity: 0.5;
        margin: 2rem 0;
    }
    
    /* Alert Boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 14px;
        border-left: 4px solid #31487A;
        font-family: 'Space Grotesk', sans-serif;
        color: #192338;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #31487A;
    }
    
    /* Markdown Text Styling */
    .main p, .main li {
        font-family: 'Space Grotesk', sans-serif;
        color: #192338;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .main h1, .main h2, .main h3, .main h4 {
        font-family: 'Space Grotesk', sans-serif;
        color: #192338;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .main strong {
        color: #31487A;
        font-weight: 600;
    }
    
    /* Modern Footer */
    .footer-text {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF;
        background: linear-gradient(135deg, #1E2E4F 0%, #31487A 100%);
        padding: 2.5rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 4px 15px rgba(30, 46, 79, 0.2);
    }
    
    .footer-text p {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Loading Spinner Text */
    .stSpinner > div > div {
        color: #192338;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(217, 225, 241, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #31487A;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1E2E4F;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load and cache the chatbot (only loads once)"""
    with st.spinner("🚀 Loading astrophysics knowledge base..."):
        chatbot = AstrophysicsRAGChatbot()
    return chatbot


def main():
    # Header
    st.markdown('<p class="main-header">Astrophysics Research Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me anything about astrophysics research papers</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot helps you explore **100 astrophysics research papers** 
        using AI-powered semantic search.
        
        **Features:**
        - Semantic search across 23,320+ chunks
        - Powered by Google Gemini Pro
        - Cites original research papers
        - Explains complex concepts simply
        """)
        
        st.divider()
        
        st.header("Settings")
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources = more context but slower responses"
        )
        
        show_sources = st.checkbox("Show source citations", value=False)
        
        st.divider()
        
        st.header("Example Questions")
        example_questions = [
            "What are gamma-ray bursts?",
            "Explain dark matter in simple terms",
            "How do black holes form?",
            "What is gravitational lensing?",
            "Tell me about galaxy clusters"
        ]
        
        for question in example_questions:
            if st.button(question, key=question):
                st.session_state.current_question = question
        
        st.divider()
        
        # Stats
        st.header("Statistics")
        st.metric("Research Papers", "100")
        st.metric("Text Chunks", "23,320")
        st.metric("Total Tokens", "14.8M")
    
    # Initialize chatbot
    try:
        chatbot = load_chatbot()
        
        # Update top_k if changed
        chatbot.top_k = top_k
        
    except Exception as e:
        st.error(f"Failed to load chatbot: {str(e)}")
        st.info("Make sure you've run `py create_embeddings.py` first!")
        st.stop()
    
    # Main chat interface
    st.divider()
    
    # Question input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Ask your question:",
            placeholder="e.g., What are the properties of dark matter halos?",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Handle example question from sidebar
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
        ask_button = True
    
    # Process question
    if ask_button and question:
        # Display user question
        with st.chat_message("user"):
            st.markdown(f"**{question}**")
        
        # Generate response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for chunk in chatbot.stream_ask(question, include_sources=show_sources):
                    full_response += chunk
                    response_placeholder.markdown(full_response)
                    time.sleep(0.01)  # Small delay for smooth streaming effect
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("This might be due to:")
                st.markdown("""
                - FAISS index not created yet (run `py create_embeddings.py`)
                - Invalid Gemini API key in `.env` file
                - Network connection issues
                """)
    
    elif ask_button and not question:
        st.warning("Please enter a question first!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div class='footer-text'>
        <p style='margin: 0; font-size: 0.95rem;'>Powered by Google Gemini Pro, FAISS, and Sentence Transformers</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;'>Based on 100 astrophysics research papers from arXiv</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
