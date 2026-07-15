# AI Platform — Panduan Penggunaan

## Apa itu AI Platform?

AI Platform adalah sistem AI end-to-end yang berjalan secara on-premise di laptop. Platform ini mencakup:

1. **Model Serving** — Ollama menjalankan model LLM seperti Phi-3, Qwen2.5, dan Gemma3
2. **AI Router** — LiteLLM menyediakan API OpenAI-compatible untuk routing ke berbagai model
3. **Observability** — Langfuse mencatat setiap panggilan LLM untuk monitoring dan debugging
4. **RAG Pipeline** — LlamaIndex + Qdrant untuk tanya jawab berbasis dokumen
5. **Agent Framework** — LangGraph untuk workflow agent multi-step

## Cara Menggunakan

### Chat Sederhana
Kirim request ke endpoint `/v1/chat/completions` dengan format OpenAI-compatible.

### Structured Output
Gunakan endpoint `/v1/structured` untuk menghasilkan JSON terstruktur sesuai schema.

### RAG (Retrieval-Augmented Generation)
1. Upload dokumen via `/v1/rag/ingest`
2. Tanyakan pertanyaan via `/v1/rag/query`
3. Sistem akan mencari konteks relevan dari dokumen dan menghasilkan jawaban dengan sitasi.

## Teknologi yang Digunakan
- **Backend**: Python, FastAPI
- **LLM**: Ollama (Phi-3 Mini, Qwen2.5 3B)
- **Router**: LiteLLM
- **Vector DB**: Qdrant
- **RAG Framework**: LlamaIndex
- **Database**: PostgreSQL
- **Cache**: Redis
- **Observability**: Langfuse

## Spesifikasi Minimal
- RAM: 8 GB (16 GB recommended)
- CPU: 4 core (Intel atau Apple Silicon)
- Storage: SSD 512 GB
- GPU: Tidak diperlukan
