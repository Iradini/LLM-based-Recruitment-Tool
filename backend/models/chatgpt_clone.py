from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from backend.llm_factory import get_llm
from backend.config import settings


class ChatAssistant:
    def __init__(self, llm_model, api_key, temperature=0, history_length=3):
        """
        Initialize the ChatAssistant class.

        Parameters
        ----------
        llm_model : str
            The model name.

        api_key : str
            The API key for accessing the LLM model.

        temperature : float
            The temperature parameter for generating responses.

        history_length : int, optional
            The length of the conversation history to be stored in memory. Default is 3.
        """
        
        template = (
            "The following is a conversation between a human and an AI assistant.\n\n"
            "Chat history:\n{history}\n\n"
            "Human: {human_input}\n"
            "AI assistant:"
        )
       
        self.prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template,
        )
        self.llm = get_llm(
            model=llm_model,
            api_key=api_key,
            temperature=temperature,
        )
        self.model = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=ConversationBufferWindowMemory(k=history_length),
            output_key="output",
            verbose=settings.LANGCHAIN_VERBOSE,
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
        response = self.model.invoke(human_input)

        return response


if __name__ == "__main__":
    # Create an instance of ChatAssistant with appropriate settings
    chat_assistant = ChatAssistant(
        llm_model=settings.GEMINI_LLM_MODEL,
        api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        history_length=2,
    )

    # Use the instance to generate a response
    output = chat_assistant.predict(
        human_input="what is the answer to life the universe and everything?"
    )

    print(output)
