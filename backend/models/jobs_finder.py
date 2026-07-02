from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from backend.llm_factory import get_llm
from backend.config import settings
from backend.models.resume_summarizer_chain import get_resume_summarizer_chain
from backend.retriever import Retriever

resume_summarizer = get_resume_summarizer_chain()

def format_jobs(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        formatted.append(
            f"Job {i}:\n"
            f"Title: {meta.get('title', 'N/A')}\n"
            f"Company: {meta.get('company', 'N/A')}\n"
            f"Location: {meta.get('location', 'N/A')}\n"
            f"Employment Type: {meta.get('employment_type', 'N/A')}\n"
            f"Seniority: {meta.get('seniority_level', 'N/A')}\n"
            f"URL: {meta.get('post_url', 'N/A')}\n"
            f"Description excerpt: {doc.page_content}\n"
        )
    return "\n".join(formatted)


class JobsFinderAssistant:
    def __init__(
        self, resume, llm_model, api_key, temperature=0, history_length=3
    ):
        """
        Initialize the JobsFinderAssistant class.

        Parameters
        ----------
        resume : str
            The resume of the user.

        llm_model : str
            The model name.

        api_key : str
            The API key for accessing the LLM model.

        temperature : float
            The temperature parameter for generating responses.

        history_length : int, optional
            The length of the conversation history to be stored in memory. Default is 3.
        """
        # Make a summary of the resume for the queries
        # Use resume_summarizer_chain.
        self.resume = resume
        result = resume_summarizer.invoke({"resume": resume})
        self.resume_summary = result["text"]

        # Initialize the jobs retriever
        self.retriever = Retriever()

        template = (
            "You are a job matching assistant. Your ONLY job is to present the job postings "
            "retrieved below and explain how they relate to the candidate's resume.\n\n"
            "STRICT RULES:\n"
            "- Only reference jobs that appear in the 'Relevant job postings' section below. "
            "Do not suggest, invent, or recall any jobs from outside this list.\n"
            "- Only reference skills, experience, or qualifications that appear explicitly "
            "in the resume below. Do not infer or assume anything about the candidate.\n"
            "- If the retrieved jobs are a poor match, say so honestly rather than "
            "inventing reasons they might fit.\n"
            "- If you cannot answer from the provided data, say: "
            "'I don't have enough information to answer that from the current search results.'\n\n"
            "Resume:\n{resume}\n\n"
            "Chat history:\n{history}\n\n"
            "Relevant job postings retrieved:\n{search_results}\n\n"
            "Human: {human_input}\n"
            "AI assistant:"
        )

        self.prompt = PromptTemplate(
            input_variables=["resume", "history", "search_results", "human_input"],
            template=template,
        )
        self.llm = get_llm(
            model=llm_model,
            api_key=api_key,
            temperature=temperature,
        )

        # Create a memory for the chat assistant.
        _memory = ConversationBufferWindowMemory(
            input_key="human_input", k=history_length
        )

        self.model = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=_memory,
            output_key="output",
        )

    def predict(self, human_input: str) -> str:
        """
        Generate a response to a human input.

        Parameters
        ----------
        human_input : str
            The human input to the chat assistant.

        Returns
        -------
        response : str
            The response from the chat assistant.
        """

        
        jobs = self.retriever.search(human_input + " " + self.resume_summary)
        formatted_jobs = format_jobs(jobs)

        model_answer = self.model.invoke(
            {"resume": self.resume, "search_results": formatted_jobs, "human_input": human_input}
        )

        return model_answer


if __name__ == "__main__":
    # Create an instance of JobFinderAssistant with appropriate settings
    resume = """
John Doe
john.doe@email.com

Objective:

Results-driven and highly skilled Software Engineer with 5 years of experience designing, developing, and maintaining cutting-edge software solutions. Adept at collaborating with cross-functional teams to drive project success.

Education:

Bachelor of Science in Computer Science
ABC University, Anytown, USA
Graduation Date: May 2020

Technical Skills:

Programming Languages: Java, Python, JavaScript
Web Technologies: HTML5, CSS3, React.js
Database Management: MySQL, MongoDB
Frameworks and Libraries: Spring Boot, Node.js
Version Control: Git
Operating Systems: Windows, Linux
Other Tools: JIRA, Docker

Professional Experience:

Software Engineer | XYZ Tech, Anytown, USA | June 2020 - Present

Developed and maintained scalable web applications using Java and Spring Boot, resulting in a 15% improvement in application performance.
Conducted code reviews and provided constructive feedback to team members, resulting in improved code quality and adherence to coding standards.
Participated in agile development processes, including daily stand-ups, sprint planning, and retrospective meetings.

Projects:

E-commerce Platform Redesign | March 2022 - June 2022

Led the redesign of the e-commerce platform using React.js, resulting in a 20% increase in user engagement and a 15% improvement in page load times.
Inventory Management System | September 2019 - December 2019

Developed a robust inventory management system using Java and Spring Boot, streamlining the tracking of product stock and reducing errors by 30%.

Certifications:

Oracle Certified Professional, Java SE Programmer

Professional Memberships:

Member, Association for Computing Machinery (ACM)
"""
    chat_assistant = JobsFinderAssistant(
        resume=resume,
        llm_model=settings.GEMINI_LLM_MODEL,
        api_key=settings.GOOGLE_API_KEY,
        temperature=0,
    )

    # Use the instance to generate a response
    output = chat_assistant.predict(
        human_input="I'm looking for a job as a software engineer."
    )

    print(output["output"])
