---
title: Mistral 7b API
emoji: 🐢
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# Mistral_7B + RAG API #

This project consists on an API hosted on HuggingFace Spaces that generates text using the amazing Mistral 7B model.

## Tech stack:
- LangChain: LLM orchestration, document loader, splitter, RAG
- FastAPI: API end-points
- HuggingFace: LLM, Embeddings, API hosting
- ChromaDB: Vector DB

## Parameters (best combination):
- Text splitter: RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=200)
- LLM: mistralai/Mistral-7B-Instruct-v0.1("temperature":0.1, "max_new_tokens":300)
- Retriever: similarity search, top k=4, "score_threshold": .95
- chain_type: stuff
