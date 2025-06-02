# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) database project for NBER working papers. The goal is to create a searchable database from NBER working paper PDFs using Python.

## Repository Structure

- `code/` - Python RAG implementation using LlamaIndex
  - `ingest_pdfs.py` - PDF ingestion script to create vector database
  - `query_rag.py` - Query interface for searching papers
- `data/pdf/` - Sample NBER working papers (w32412.pdf through w32436.pdf)
- `chroma_db/` - ChromaDB vector store (created after ingestion)
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variable template

## Setup and Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your OpenAI API key
3. Ingest PDFs: `cd code && python ingest_pdfs.py`
4. Query database: `cd code && python query_rag.py`

## Development Notes

- Uses LlamaIndex framework for RAG implementation
- ChromaDB for vector storage with persistent storage
- OpenAI embeddings (text-embedding-3-small) and GPT-3.5-turbo for LLM
- Sample data consists of 25 NBER working papers in PDF format