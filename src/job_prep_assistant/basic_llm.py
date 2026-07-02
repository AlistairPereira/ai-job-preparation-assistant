"""Step 0: Basic LLM call using LangChain, Ollama, and Llama 3.2."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
# ChatOllama is the LangChain class that connects our Python code to Ollama.
# Llama 3.2 = the actual AI model
# ChatOllama = LangChain connector used to communicate with Ollama
#and Ollama runs llama3.2


def create_model() -> ChatOllama:
    """Create and return the local Ollama model."""

    return ChatOllama(
        model="llama3.2:latest",
        temperature=0,
    )
# Create a LangChain connection to Ollama
# and tell Ollama to use the llama3.2 model.

def main() -> None:
    """Take a user question and send it to the local LLM."""

    model = create_model()

    user_question = input(
        "Ask a job preparation question: "
    ).strip()

    if not user_question:
        print("Please enter a question.")
        return

    messages = [
        SystemMessage(
            content=(
                "You are a helpful job preparation assistant. "
                "Explain answers clearly, accurately, and simply."
            )
        ),
        HumanMessage(content=user_question),
    ]

    print("\nSending question to Llama 3.2...\n")

    response = model.invoke(messages)

    print("AI Response:\n")
    print(response.content)


if __name__ == "__main__":
    main()