"""Step 1: One job-preparation agent with one calculation tool."""

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def calculate_skill_match(
    matched_skills: int,
    total_required_skills: int,
) -> dict[str, Any]:
    """
    Calculate the percentage of job-required skills that a candidate matches.

    Args:
        matched_skills: Number of required skills the candidate has.
        total_required_skills: Total number of skills required by the job.
    """

    if total_required_skills <= 0:
        return {
            "error": "Total required skills must be greater than zero."
        }

    if matched_skills < 0:
        return {
            "error": "Matched skills cannot be negative."
        }

    if matched_skills > total_required_skills:
        return {
            "error": (
                "Matched skills cannot be greater than "
                "the total required skills."
            )
        }

    match_percentage = (
        matched_skills / total_required_skills
    ) * 100

    return {
        "matched_skills": matched_skills,
        "total_required_skills": total_required_skills,
        "match_percentage": round(match_percentage, 2),
    }


def create_skill_match_agent():
    """Create and return the job skill match agent."""

    model = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
    )
    
# ChatOllama connects LangChain to Ollama
#         ↓
# Ollama is told to use llama3.2

# The parts are:
# ChatOllama = LangChain connector
# Ollama = runs the model locally
# Llama 3.2 = actual AI model

    agent = create_agent(
        model=model,
        tools=[calculate_skill_match],
        system_prompt=(
            "You are a helpful job preparation assistant. "
            "When the user asks for a job skill match percentage, "
            "always use the calculate_skill_match tool. "
            "Do not calculate the percentage yourself. "
            "After receiving the tool result, explain it simply."
        ),
    )

    return agent


def main() -> None:
    """Take a user request and run the skill match agent."""

    agent = create_skill_match_agent()

    user_request = input(
        "Ask about your job skill match: "
    ).strip()

    if not user_request:
        print("Please enter a request.")
        return

    print("\nRunning agent...\n")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_request,
                }
            ]
        }
    )

    print_agent_steps(result)

    final_message = result["messages"][-1]

    print("\nFinal Answer:\n")
    print(final_message.content)


def print_agent_steps(result: dict[str, Any]) -> None:
    """Display the tool selected by the model and its result."""
    print("testing")
    for message in result["messages"]:
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for tool_call in tool_calls:
                print(f"Tool selected: {tool_call['name']}")
                print(f"Tool arguments: {tool_call['args']}")

        if getattr(message, "type", "") == "tool":
            print(f"Tool result: {message.content}")


if __name__ == "__main__":
    main()