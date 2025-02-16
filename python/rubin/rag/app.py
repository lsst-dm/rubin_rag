import streamlit as st
from chatbot import configure_retriever, create_qa_chain, handle_user_input
from dotenv import load_dotenv
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from layout import setup_header_and_footer, setup_landing_page, setup_sidebar
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Set page configuration and design
st.set_page_config(
    page_title="Vera - Rubin Observatory Bot",
    initial_sidebar_state="collapsed",
    page_icon="../../../static/rubin_avatar_color.png",
)
st.logo("../../../static/logo.png")

# Load the CSS file
with open("../../../static/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Set up the session state
if "message_sent" not in st.session_state:
    st.session_state.message_sent = False

# Configure the Weaviate retriever and QA chain
retriever = configure_retriever()
qa_chain = create_qa_chain(retriever)

# Enable dynamic filtering based on user input
setup_sidebar()

# Set up the landing page
setup_landing_page()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

# Set up the header and footer
setup_header_and_footer(msgs)

# Handle user input and chat history
handle_user_input(qa_chain, msgs)
