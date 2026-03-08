<p align="center">
  <img src="/public/thumbnail.png" alt="Thumbnail" width="800">
</p>

## BBUmate - RAG-based AI Policy Assistant

description: Chatbot Counseling Service for Newlyweds

### 1. Service Overview

- **Purpose**: "BBUmate" is an AI counseling platform that integrates information scattered across various agencies — such as housing, loans, welfare, and corporate benefits — using RAG technology, and recommends policies tailored to the user's situation.
- **User Flow**: Users input conditions such as income, region, number of children, etc. → The system searches for relevant policies/loans/subsidies → Summarizes and compares results → Provides **answers with cited sources**.

### 2. Project Motivation

1. **Fragmented Information**: Users must visit multiple sites (Housing & Urban Fund, Ministry of Health and Welfare, MyHome Portal, etc.) to gather information, and inconsistent terminology makes it difficult to obtain accurate, up-to-date understanding.
2. **Integrated Counseling via AI**: Public data APIs, policy documents, and official announcements are stored in a vector DB using RAG. Answers are safely generated based on documents matched to queries. **Source documents and references** such as official notices, press releases, and public data links are presented alongside answers to enhance credibility.
3. **Key Features** (v1.0)

- AI chatbot counseling
- Source-based evidence presentation

### 3. Architecture / Design (LangChain + Chroma + Upstage)

- **FastAPI**: Lightweight REST API server (`/` health check, `/query` query endpoint)
- **LangChain 0.3 (Runnables API)**: Chain composition
  - Retriever(Chroma) → Context formatting → Prompt → Upstage Chat → String parsing
- **Vector DB: Chroma**: Local persistence (`CHROMA_DB_DIR`)
- **Model**: Upstage Embedding/Chat models managed via `.env`
- **Processing Flow**
  1. Question (User question input)
  2. Clarification Check (Determine question clarity)
  3. Re-ask / Clarification (Re-question or refine the query)
  4. Retrieve (Search related documents from VectorDB)
  5. Grade (Evaluate document relevance and re-rank)
  6. Re-write Query (Query rewriting)
  7. Web Search (External data search)
  8. LLM Answer Generation (Generate answer using Upstage Chat model)
  9. Answer (Output)

### 4. RAG Pipeline

<p align="center">
  <img src="/public/rag_pipeline.png" alt="RAG Pipeline Diagram" width="800">
</p>

### 5. Folder Structure

```text
KDT_BE13_TOY_PROJECT4/
├── data/                 # PDF and text source files
├── scripts/              # Crawling and preprocessing scripts
├── src/
│   ├── ingestion/        # Collection → Chunking → Embedding → Storage
│   ├── retrieval/        # Query-based document retrieval
│   ├── generation/       # LLM response generation
│   ├── chains/           # LangChain / LangGraph chain definitions
│   ├── utils/            # Common utilities, logging, caching
│   └── api/              # FastAPI endpoints
├── .env.example
├── requirements.txt
├── main.py
└── README.md
```

### 6. Tech Stack

- **Language**: Python 3.13
- **Web**: FastAPI, Uvicorn
- **AI Orchestration**: LangChain 0.3.x, langchain-community 0.3.x, langchain-upstage
- **Vector DB**: ChromaDB
- **LLM/Embedding**: Upstage Solar Series

### 7. Environment Variables (.env)

Create a `.env` file in the project root and set the following keys:

```dotenv
UPSTAGE_API_KEY=YOUR_UPSTAGE_API_KEY
UPSTAGE_EMBEDDING_MODEL=solar-embedding-1-large
UPSTAGE_CHAT_MODEL=solar-1-mini-chat
CHROMA_DB_DIR=./chroma_storage
```

- `UPSTAGE_EMBEDDING_MODEL` / `UPSTAGE_CHAT_MODEL`: You can swap model names to experiment as needed.
- `CHROMA_DB_DIR`: Path to the Chroma persistence directory.

### 8. Installation & Running

1. Create/activate a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate

# PowerShell
.\venv\Scripts\Activate.ps1
```

2. Install packages (version pinned: LangChain 0.3)

```bash
pip install -r requirements.txt
```

3. Create a `.env` file (refer to the example above)

4. Verify the `.env` file

```bash
cat .env
```

5. **Build the Vector DB (Important: Must be done before starting the server)**

```bash
python run_ingestion.py
```

This step **must be executed** during deployment. RAG search will not function without the vector DB.

6. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 9. API Usage

- **Health Check**: `GET /`
- **Query**: `POST /query`
  - Request Body
    ```json
    { "question": "Tell me about newlywed housing loan requirements" }
    ```
  - Response Example
    ```json
    { "answer": "...model response string..." }
    ```

### 10. Development / Operations Tips

- Select the project's **virtual environment interpreter** in your IDE (e.g., `venv/bin/python`).
- If you modify `.env`, it is recommended to **restart the server** or use the `--reload` option.
- If the Chroma index is empty, search will not work properly. Make sure to perform embedding/ingestion beforehand.

---

This service aims to safely provide newlywed-tailored policy information based on reliable sources.
