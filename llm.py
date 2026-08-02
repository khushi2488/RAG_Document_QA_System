"""
Step 7:
Load the Groq LLM.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.1


# -------------------------------------------------------
# Load LLM
# -------------------------------------------------------

def load_llm():
    """
    Load the Groq LLM.
    """

    print("=" * 60)
    print("Loading Groq LLM")
    print("=" * 60)

    # Load .env
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file."
        )

    print("Connecting to Groq...")

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=api_key,
    )

    print("Groq LLM Connected.")

    return llm


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    llm = load_llm()

    response = llm.invoke("What is Artificial Intelligence?")

    print("\nResponse:\n")
    print(response.content)