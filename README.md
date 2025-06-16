#  RAG-LLM: Document-Aware Q&A API with OpenAI, LangChain & FastAPI
RAG-LLM is a Retrieval-Augmented Generation (RAG) system that allows users to query their documents using natural language. Powered by OpenAI LLMs, LangChain, and FastAPI, this tool reads documents like PDFs and DOCX files, retrieves relevant context, and generates intelligent answers. It also supports multi-turn conversation memory.

 FEATURES
📄 Document Parsing: Supports loading .pdf, .docx, and other text-based formats.
🔗 LangChain + LCEL: Uses LangChain Expression Language for building modular chains.
🧠 Conversation Memory: Manages memory to support multi-turn Q&A interactions.
⚡ FastAPI Server: Exposes endpoints to interact with the agent via RESTful APIs.
🤖 OpenAI LLMs: Uses GPT-3.5/4 for response generation.
📦 Embeddings + Vector Store: Retrieves relevant chunks using  Chroma.

🛠️ How It Works
Document Ingestion: PDFs, DOCX files are split into chunks and embedded.
Storage: Embeddings are stored in a vector store (e.g. FAISS).
Query: Relevant chunks are retrieved and passed to OpenAI's LLM with your question.
Response: A coherent, context-aware answer is returned via the FastAPI endpoint.
Memory: Maintains history of user queries to support follow-up questions.



## 📦 Quickstart

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/rag-llm.git
cd rag-llm

2. ** Install Requirements ** :

```bash
pip install -r requirements.txt

3. ** Set Environment Variables** :

```bash
OPENAI_API_KEY=your_openai_key_here
