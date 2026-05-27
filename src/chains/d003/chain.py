"""7. Chain: build the LLM chain"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.retrieval.d003.retriever import get_retriever
from src.generation.d003.prompting import build_chat_prompt, format_docs_for_context


def build_llm() -> ChatOpenAI:
    """
    Initialize OpenAI Chat model from environment variables.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_CHAT_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment")
    if not model_name:
        raise ValueError("OPENAI_CHAT_MODEL is not set in environment")

    return ChatOpenAI(api_key=api_key, model=model_name)


def build_chain(k: int = 3):
    """
    Build a RAG chain: retriever -> context formatter -> prompt -> LLM -> string output.
    """
    retriever = get_retriever(k=k)
    prompt = build_chat_prompt()
    llm = build_llm()

    chain = (
        {
            "context": retriever | format_docs_for_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def answer_question(question: str, k: int = 3) -> Tuple[str, List[Document]]:
    """
    Retrieve top-k documents, run the chain, and return (answer, docs).
    """
    retriever = get_retriever(k=k)
    docs: List[Document] = retriever.invoke(question)

    prompt = build_chat_prompt()
    llm = build_llm()

    chain = (
        {
            "context": lambda x: format_docs_for_context(x["context"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"question": question, "context": docs})
    return answer, docs
