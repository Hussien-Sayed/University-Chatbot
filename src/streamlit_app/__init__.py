"""Streamlit app module."""

from .chat import chat_page
from .evaluation import evaluation_page
from .settings import settings_page
from .dashboard.dashboard_page import dashboard_page

__all__ = ['chat_page', 'evaluation_page', 'settings_page', 'dashboard_page']
