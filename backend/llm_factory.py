from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import settings


def get_llm(temperature=0, model=None, api_key=None, **kwargs):
    """
    Factory function to get a Gemini LLM instance.

    Parameters
    ----------
    temperature : float
        The temperature parameter for generating responses.
    model : str, optional
        Override the default model from settings.
    api_key : str, optional
        Override the default API key from settings.

    Returns
    -------
    llm : ChatGoogleGenerativeAI
        The Gemini LLM instance.
    """
    return ChatGoogleGenerativeAI(
        model=model or settings.GEMINI_LLM_MODEL,
        google_api_key=api_key or settings.GOOGLE_API_KEY,
        temperature=temperature,
    )
