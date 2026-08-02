"""
Step 8:
Build the complete RAG Chain.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from retrieve import (
    load_embedding_model,
    connect_qdrant,
    create_vectorstore,
)

from prompt import build_prompt
from llm import load_llm


# -------------------------------------------------------
# Initialize Components
# -------------------------------------------------------

def initialize_rag():

    embeddings = load_embedding_model()

    client = connect_qdrant()

    vectorstore = create_vectorstore(
        client,
        embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = load_llm()

    return retriever, llm, client


# -------------------------------------------------------
# Format Retrieved Documents
# -------------------------------------------------------

def format_docs(docs):
    """
    Convert retrieved documents into one string.
    """

    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------------------------------------
# Prompt Function
# -------------------------------------------------------

def create_prompt(inputs):
    """
    Build the final prompt.
    """

    return build_prompt(
        question=inputs["question"],
        documents=inputs["context"],
    )


# -------------------------------------------------------
# Build LangChain RAG Chain
# -------------------------------------------------------

def build_rag_chain():

    retriever, llm, client = initialize_rag()

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | create_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, client


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    rag_chain, client = build_rag_chain()

    print("=" * 60)
    print("RAG Chatbot Ready")
    print("=" * 60)

    while True:

        question = input("\nAsk Question : ")

        if question.lower() == "exit":
            break

        print("\nSearching...\n")

        answer = rag_chain.invoke(question)

        print("=" * 60)
        print("Answer")
        print("=" * 60)
        print(answer)

    client.close()


# -------------------------------------------------------

if __name__ == "__main__":
    main()