from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from backend.config import settings
from backend.llm_factory import get_llm

# TODO: Create a string template for this chain. It must indicate the LLM
# that a resume is being provided to be summarized to extract the candidates skills.
# The template must have one input variable: `resume`.
template = (
    "You are a helpful assistant for summarizing resumes. Extract the most relevant information from the resume provided and summarize it in a concise way. "
    "Focus on extracting the candidate's skills, experience and education.\n\n"
    "Resume:\n{resume}\n\n"
    "Summary:"
)


def get_resume_summarizer_chain():
    # TODO: Create a prompt template using the string template created above.
    # Hint: Use the `PromptTemplate` class.
    # Hint: Don't forget to add the input variables: `resume`.
    prompt = PromptTemplate(
        input_variables=["resume"],
        template=template
    )


    # TODO: Create an instance of an LLM using the `get_llm` factory function with the appropriate settings.
    # Hint: You need to pass `temperature` parameter.
    llm = get_llm(temperature=0,)

# TODO: Create an instance of `LLMChain` with the appropriate settings.
    # This chain must combine our prompt and an llm. It doesn't need a memory.
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
