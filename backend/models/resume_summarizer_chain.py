from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from backend.config import settings
from backend.llm_factory import get_llm


template = (
    "You are a helpful assistant for summarizing resumes. Extract the most relevant information from the resume provided and summarize it in a concise way. "
    "Focus on extracting the candidate's skills, experience and education.\n\n"
    "Resume:\n{resume}\n\n"
    "Summary:"
)


def get_resume_summarizer_chain():

    prompt = PromptTemplate(
        input_variables=["resume"],
        template=template
    )
    llm = get_llm(temperature=0,)

    resume_summarizer_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    return resume_summarizer_chain


if __name__ == "__main__":
    resume_summarizer_chain = get_resume_summarizer_chain()
    print(
        resume_summarizer_chain.invoke(
            {"resume": "I am a software engineer with 5 years of experience"}
        )
    )
