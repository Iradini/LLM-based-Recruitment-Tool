from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.models.chatgpt_clone import ChatAssistant


def test_structure_chat_assistant():
    chat_assistant = ChatAssistant(
        llm_model=settings.GEMINI_LLM_MODEL,
        api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        history_length=2,
    )

    # ChatAssistant attributes
    assert hasattr(chat_assistant, "prompt")
    assert hasattr(chat_assistant, "llm")
    assert hasattr(chat_assistant, "model")
    assert callable(chat_assistant.predict)

    # ChatAssistant attribute types - support both providers
    assert isinstance(chat_assistant.llm, ChatGoogleGenerativeAI)
    assert isinstance(chat_assistant.prompt, PromptTemplate)
    assert isinstance(chat_assistant.model, LLMChain)

    # ChatAssistant model attribute types
    assert isinstance(
        chat_assistant.model.memory, ConversationBufferWindowMemory
    )

    # ChatAssistant model attribute values
    assert chat_assistant.model.llm == chat_assistant.llm
    assert chat_assistant.model.llm.temperature == 0
    assert chat_assistant.model.prompt == chat_assistant.prompt
    assert chat_assistant.model.verbose == settings.LANGCHAIN_VERBOSE
    assert chat_assistant.model.memory.k == 2
    assert chat_assistant.model.memory.k == 2
