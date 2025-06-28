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

---

## 🗄️ Datasets

| Dataset                                                                | Task                       | Pubyear | Documents                                                         | Questions                                                                  | Answers                                                 | Metrics                                     |
| :--------------------------------------------------------------------- | :------------------------- | :------ | :---------------------------------------------------------------- | :------------------------------------------------------------------------- | :------------------------------------------------------ | :------------------------------------------ |
| [Natural Questions (NQ)](./benchmarks/NQ.md)                           | Factoid QA                 | 2019    | Wikipedia                                                         | 323,045 questions with each an wikipedia page                              | paragraph/span                                          | Rouge, EM                                   |
| [TriviaQA](./benchmarks/TriviaQA.md)                                   | Factoid QA                 | 2017    | 662,659 evidence documents                                        | 95,956 QA pairs                                                            | text string (92.85% wikipedia titles)                   | EM                                          |
| [NarrativeQA (NQA)](./benchmarks/NarrativeQA.md)                       | Factoid QA                 | 2017    | 1,572 stories (books,movie scripts) & human generated summaries | 46,765 human generated questions                                           | human written, short, averaging 4.73 tokens             | Rouge                                       |
| [SQuAD](https://huggingface.co/datasets/rajpurkar/squad)               | Factoid QA                 | 2016    | 536 articles                                                      | 107,785 question-answer pairs                                              | spans                                                   | EM                                          |
| [PopQA](./benchmarks/PopQA.md)                                         | Factoid QA                 | 2023    | wikipedia                                                         | 14k questions                                                              | long-tail entites                                       | EM                                          |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)           | Factoid QA                 | 2019    | 25k Activity contexts and 45k WikiHow contexts                    | 70k examples                                                               | classification                                          | Accuracy                                    |
| [StrategyQA](https://allenai.org/data/strategyqa)                      | Factoid QA                 | 2021    | wikipedia (1,799 Wikipedia terms)                                 | 2,780 strategy questions                                                   | its decomposition, evidence paragraphs                  | EM                                          |
| [Fermi](https://allenai.org/data/fermi)                                | Factoid QA                 | 2021    | -                                                                 | 928 FPs (a question Q, an answer A, supporting facts F, an explanation P)  | spans                                                   | Accuracy                                    |
| [2WikiMultihopQA](./benchmarks/2WikiMHQA.md)                           | Multi-Hop QA               | 2020    | articles from wikipedia and wikidata                              | 192,606 questions each with a context                                      | textual spans, sentence-level supporting facts, evidence (tiples) | F1                                          |
| [HotpotQA](/benchrag/hotpotqa)                                      | Multi-Hop QA               | 2018    | The whole wikipedia dump                                          | 112,779 question-answer pairs                                              | text span                                               | F1                                          |
| [BRIGHT](/benchrag/bright)                           | Long-Form QA               | 2025    | -     | 12 tasks, each ~100 questions    | multiple sentences                                      | NDCG@10, LLMScore |
| [ELI5](https://huggingface.co/datasets/eli5)                           | Long-Form QA               | 2019    | 250 billion pages from Common Crawl                               | 272K questions                                                             | multiple sentences                                      | Citation Recall, Citation Precision, Claim Recall |
| [WikiEval](https://huggingface.co/datasets/explodinggradients/WikiEval) | Long-Form QA               | 2023    | 50 wikipedia pages                                                | 50 questions                                                               | text spans (sentences)                                  | Ragas                                       |
| [ASQA](./benchmarks/ASQA.md)                                           | Long-Form QA               | 2022    | wikipedia                                                         | 6,316 ambiguous factoid questions                                          | long-form answers                                       | disambig F1, RougeL, EM                     |
| [WebGLM-QA](https://huggingface.co/datasets/THUDM/webglm-qa)           | Long-Form QA               | 2023    | -                                                                 | 44979 samples                                                              | sentences                                               | RougeL, Citation Recall, Citation Precision |
| [TruthfulQA](https://huggingface.co/datasets/truthful_qa)              | Multiple Choice QA         | 2021    | -                                                                 | 817 questions that span 38 categories                                      | sentence answer/multiple choice                         | EM                                          |
| [MMLU](https://huggingface.co/datasets/cais/mmlu)                      | Multiple Choice QA         | 2021    | -                                                                 | 15,908 multiple-choice questions                                           | 4-way multiple choice                                   | Accuracy                                    |
| [OpenBook QA](https://huggingface.co/datasets/allenai/openbookqa)      | Multiple Choice QA         | 2018    | 7326 facts from a book                                            | 5,957 questions                                                            | 4-way multiple-choice                                   | Accuracy                                    |
| [QuALITY (QLTY)](https://github.com/nyu-mll/quality)                   | Multiple Choice QA         | 2022    | -                                                                 | 6,737 questions                                                            | 4-way multiple choices                                  | Accuracy                                    |
| [WikiAsp](https://huggingface.co/datasets/wiki_asp)                    | Open-Domain Summarization  | 2021    | Wikipedia articles from 20 different domains                      | 320,272 samples                                                            | 1) aspect selection (section title), 2) summary generation (section paragraph) | ROUGE, F1, UniEval                          |
| [Scifact](https://huggingface.co/datasets/BeIR/scifact)                | Fact-checking              | 2020    | 5,183 abstracts                                                   | 1409 claim-abstract pairs                                                  | 3-class classification (support/refutes/Noinfo)         | nDCG@10                                     |
| [FEVER](https://huggingface.co/datasets/fever)                         | Fact-checking              | 2018    | 50,000 popular pages from wikipedia                               | 185,445 claims                                                             | 3-class classification                                  | Accuracy                                    |
| [Feverous](https://huggingface.co/datasets/fever/feverous)             | Fact-checking              | 2021    | wikipedia                                                         | 87,026 claims                                                              | 3-class classification/evidence retrieval               | Accuracy                                    |