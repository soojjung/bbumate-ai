from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from langchain_core.documents import Document
from dotenv import load_dotenv


class VectorStoreManager:
    @staticmethod
    def save_documents(
        documents: list[Document],
        collection_name=None,
        db_path=None,
        api_key=None,
        embedding_model=None,
        batch_size=3,
    ):
        """문서를 벡터 저장소에 저장"""

        load_dotenv()

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        db_path = db_path or os.getenv("CHROMA_DB_DIR", "./chroma_storage")
        embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        collection_name = collection_name or os.getenv(
            "COLLECTION_NAME", "pdf_subscription_chunks"
        )

        # OpenAI Embeddings 초기화
        embeddings = OpenAIEmbeddings(api_key=api_key, model=embedding_model)

        # from_documents를 사용하여 벡터 저장소 생성 및 저장
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=db_path,
        )
        print(f"ChromaDB 저장 완료: {db_path}")

        # 저장된 데이터 확인
        collection_data = vectorstore.get()
        print(f"저장된 문서 수: {len(collection_data['ids'])}")

        return vectorstore
