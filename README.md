# 🔬 Pharma-Intelligence AI Agent
> **Self-Hosted, Open-Source Multi-Agent RAG System for Medical Intelligence**

An enterprise-grade, open-source AI agentic pipeline designed to process high-volume biomedical literature, clinical datasets, and regulatory documentation locally without third-party API dependencies (No OpenAI, No Pinecone Cloud).

---

## 🛠️ Open-Source Technical Stack
- **LLM Serving & Inference Engine:** Ollama / llama.cpp (GGUF Quantization Q4_K_M) & vLLM
- **Model Family:** Llama-3-8B-Instruct, Mistral-7B-Instruct, Qwen2.5
- **RAG & Vector Storage:** LangChain, LlamaIndex, ChromaDB, FAISS
- **Multi-Agent Orchestration:** Custom Agentic Workflows (Tool Calling, Multi-Step Reasoning)
- **Search & Front-End Integration:** SearXNG (Web Search) & LibreChat UI
- **Environment:** Ubuntu Linux 22.04 LTS, Python 3.10, Docker, NVIDIA CUDA 12.x

---

## 📐 System Architecture

1. **Data Ingestion & Chunking:** Processes raw biomedical literature, PDF manuals, and structured CSVs into overlapping 512-token chunks.
2. **Indexing & Vector Storage:** Generates dense embeddings stored in self-hosted ChromaDB and FAISS vector stores.
3. **Retrieval & Search Augmentation:** Performs hybrid search (semantic similarity + BM25 keyword matching) and web search integration via SearXNG.
4. **Agentic Workflow:** Coordinates multi-agent reasoning for tool execution, verification, and data aggregation.
5. **Local Model Inference:** Routes structured context to quantized GGUF models (`llama.cpp` / Ollama) on local GPU hardware.
6. **Output & Presentation:** Serves precise, factual medical summary reports via LibreChat UI.

---

## 🔑 Key Features
1. **Self-Hosted LLM Serving:** Deployed GGUF quantized models locally via Ollama/llama.cpp, optimizing context length, batch size, and VRAM memory fraction.
2. **Hybrid RAG Pipeline:** Integrated dense vector retrieval (ChromaDB) with BM25 keyword matching to index medical terminology accurately.
3. **Multi-Agent Reasoning:** Designed tool-use agents capable of executing autonomous document searches, data aggregation, and verification.
4. **Data Privacy & Zero Cloud Dependency:** 100% on-premise execution compliant with strict healthcare enterprise data requirements.

---

## 🚀 Quick Start

### 1. Model Serving (Ollama / GGUF)
```bash
# Pull and serve local model
ollama run llama3:8b-instruct-q4_K_M
