"""Main entry point for the University Chatbot Streamlit app."""
import logging
import os
import warnings

# Suppress transformers verbose logging BEFORE any imports
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Configure logging to suppress transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*__path__.*", category=UserWarning)

import streamlit as st
from dotenv import load_dotenv

from src.streamlit_app import chat_page, evaluation_page, settings_page, dashboard_page

load_dotenv()


def main():
    pg = st.navigation([
        st.Page(chat_page, title="Chat", icon="💬"),
        st.Page(dashboard_page, title="Dashboard", icon="📊"),
        st.Page(evaluation_page, title="Evaluation", icon="🧪"),
        st.Page(settings_page, title="Settings", icon="⚙️"),
    ])
    pg.run()


if __name__ == "__main__":
    main()
