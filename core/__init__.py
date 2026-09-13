"""Core logic for the Legal Document Assistant.

This package is deliberately split from the Streamlit UI (app.py) so that
every piece of business logic - parsing, chunking, prompt construction and
the AI client - can be unit tested without needing a browser or a live
API key.
"""
