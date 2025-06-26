# BenchRAG: A Modular RAG Evaluation Toolkit
A modular and extensible Retrieval-Augmented Generation (RAG) evaluation framework, including independent modules for query interpretation, retrieval, compression, and answer generation.

This project separates the RAG pipeline into four independent, reusable components:
- **Interpreter**: Understands query intent, expands or decomposes complex questions
- **Retriever**: Fetches relevant documents from a corpus
- **Compressor**: Compresses context using extractive or generative methods
- **Generator**: Generates answers based on the compressed context

---

## 🧱 Project Structure

```text
BenchRAG/
├── interpreter/ # Query understanding and expansion
├── retriever/ # BM25, dense, hybrid retrievers
├── compressor/ # LLM or rule-based compressors
├── generator/ # LLM-based answer generators
├── datasets/ # Loaders for BEIR, MTEB, HotpotQA, Bright
├── pipelines/ # Full RAG pipeline runner
├── examples/ # examples for running each component
├── requirements.txt
└── README.md
```


---

## ⚙️ Installation

```bash
git clone https://github.com/gomate-community/BenchRAG.git
cd BenchRAG
pip install -r requirements.txt
```