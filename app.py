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
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin-top: 1rem;
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
    st.markdown('<p class="main-header">🌌 Astrophysics Research Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me anything about astrophysics research papers!</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot helps you explore **100 astrophysics research papers** 
        using AI-powered semantic search.
        
        **Features:**
        - 🔍 Semantic search across 23,320+ chunks
        - 🤖 Powered by Google Gemini Pro
        - 📚 Cites original research papers
        - 💡 Explains complex concepts simply
        """)
        
        st.divider()
        
        st.header("⚙️ Settings")
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources = more context but slower responses"
        )
        
        show_sources = st.checkbox("Show source citations", value=False)
        
        st.divider()
        
        st.header("💡 Example Questions")
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
        st.header("📊 Stats")
        st.metric("Research Papers", "100")
        st.metric("Text Chunks", "23,320")
        st.metric("Total Tokens", "14.8M")
    
    # Initialize chatbot
    try:
        chatbot = load_chatbot()
        
        # Update top_k if changed
        chatbot.top_k = top_k
        
    except Exception as e:
        st.error(f"❌ Failed to load chatbot: {str(e)}")
        st.info("💡 Make sure you've run `python create_embeddings.py` first!")
        st.stop()
    
    # Main chat interface
    st.divider()
    
    # Question input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "🔭 Ask your question:",
            placeholder="e.g., What are the properties of dark matter halos?",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    # Handle example question from sidebar
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
        ask_button = True
    
    # Process question
    if ask_button and question:
        # Display user question
        with st.chat_message("user", avatar="🔭"):
            st.markdown(f"**{question}**")
        
        # Generate response with streaming
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for chunk in chatbot.stream_ask(question, include_sources=show_sources):
                    full_response += chunk
                    response_placeholder.markdown(full_response)
                    time.sleep(0.01)  # Small delay for smooth streaming effect
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 This might be due to:")
                st.markdown("""
                - FAISS index not created yet (run `python create_embeddings.py`)
                - Invalid Gemini API key in `.env` file
                - Network connection issues
                """)
    
    elif ask_button and not question:
        st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🌌 Powered by Google Gemini Pro, FAISS, and Sentence Transformers</p>
        <p>📚 Based on 100 astrophysics research papers from arXiv</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
