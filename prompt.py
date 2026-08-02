"""
Step 6:
Build the prompt using retrieved context and the user's question.
"""

# -------------------------------------------------------
# System Prompt
# -------------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question ONLY using the provided context.

Instructions:
- Use only the information present in the context.
- Do not make up facts.
- If the answer is not found in the context, reply:
  "I couldn't find the answer in the provided documents."
- Be clear, concise, and accurate.
"""


# -------------------------------------------------------
# Build Context
# -------------------------------------------------------

def build_context(documents):
    """
    Combine retrieved documents into a single context string.
    """

    context_parts = []

    for i, doc in enumerate(documents, start=1):

        context_parts.append(
            f"""
Document {i}
------------
Source : {doc.metadata.get("document")}
Chunk  : {doc.metadata.get("chunk")}

{doc.page_content}
"""
        )

    return "\n".join(context_parts)


# -------------------------------------------------------
# Build Prompt
# -------------------------------------------------------

def build_prompt(question, documents):
    """
    Create the final prompt for the LLM.
    """

    context = build_context(documents)

    prompt = f"""
{SYSTEM_PROMPT}

======================
CONTEXT
======================

{context}

======================
QUESTION
======================

{question}

======================
ANSWER
======================
"""

    return prompt


# -------------------------------------------------------
# Example
# -------------------------------------------------------

if __name__ == "__main__":

    print(
        "This module is intended to be imported into the RAG pipeline."
    )