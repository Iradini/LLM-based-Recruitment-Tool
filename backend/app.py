import sys
from pathlib import Path

# Needed for the import of config — must be before other local imports
sys.path.append(str(Path(__file__).parent.parent))

import chainlit as cl

from models.chatgpt_clone import ChatAssistant
from models.jobs_finder import JobsFinderAssistant
from models.jobs_finder_agent import JobsFinderAgent
from config import settings
from utils import extract_text_from_pdf, pdf_to_markdown


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Vanilla ChatGPT",
            markdown_description="General purpose AI assistant",
        ),
        cl.ChatProfile(
            name="Jobs finder Assistant",
            markdown_description="Find jobs matching your resume",
        ),
        cl.ChatProfile(
            name="Jobs Agent",
            markdown_description="Advanced job search with additional tools",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")

    llm_model = settings.GEMINI_LLM_MODEL
    api_key = settings.GOOGLE_API_KEY

    if chat_profile == "Vanilla ChatGPT":
        model = ChatAssistant(
            llm_model=llm_model,
            api_key=api_key,
        )
        cl.user_session.set("model", model)
        await cl.Message(content="Session started. Ask me anything!").send()
    else:
        files = await cl.AskFileMessage(
            content="Please upload your resume as PDF to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

        if files:
            file = files[0]
            resume = pdf_to_markdown(open(file.path, "rb"))

            if chat_profile == "Jobs finder Assistant":
                model = JobsFinderAssistant(
                    resume=resume,
                    llm_model=llm_model,
                    api_key=api_key,
                )
            else:
                model = JobsFinderAgent(
                    resume=resume,
                    llm_model=llm_model,
                    api_key=api_key,
                )

            cl.user_session.set("model", model)
            await cl.Message(content="Now, what kind of jobs are you looking for?").send()


@cl.on_message
async def main(message: cl.Message):
    model = cl.user_session.get("model")
    if not model:
        await cl.Message(content="Please select an assistant first!").send()
        return

    result = model.predict(message.content)["output"]
    # Gemini returns list-of-dicts instead of a plain string; unwrap it
    if isinstance(result, list):
        result = "\n".join(
            part["text"] for part in result if isinstance(part, dict) and "text" in part
        )
    await cl.Message(content=result).send()
