---
title: "BRIGHT: Benchmark for Retrieval-based Inference in QA Tasks"
layout: single
permalink: /benchrag/bright
classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "fa-solid fa-list-ul"
toc_sticky: true
author_profile: true
sidebar:
  nav:
    - docs
---


# Introduction

[BRIGHT（**B**enchmark for **R**etrieval-based **I**nference in **G**eneral **H**eterogeneous **T**exts）](https://huggingface.co/datasets/xlangai/BRIGHT) is the first text retrieval benchmark that requires intensive reasoning to retrieve relevant documents. The queries are collected from diverse domains (StackExchange, LeetCode, and math competitions), all sourced from realistic human data.

# 关键特点：

- 多子任务覆盖：包含多种问答类型（开放式、事实验证、多跳推理等）。
- 高质量文档集合：每个问题都绑定文档集合，用于检索评估。
- 支持统一评估：使用一致的评估指标（如Recall@k、MRR@k等）。


# Data Statistics

| Dataset              | Q   | 𝒟         | 𝒟⁺  | Q Len | 𝒟 Len | Q Source            | 𝒟 Source                                       |
|----------------------|-----|-----------|------|--------|--------|----------------------|------------------------------------------------|
| **StackExchange**                                                                                                          |
| Biology              | 103 | 57,364    | 3.6  | 83.6   | 115.2  | StackExchange post   | Web pages								      |
| Earth Science        | 118 | 122,388   | 7.7  | 132.4  | 113.3  | StackExchange post   | Web pages                                      |
| Economics            | 103 | 50,221    | 8.0  | 120.2  | 181.5  | StackExchange post   | Web pages                                      |
| Psychology           | 101 | 52,841    | 7.3  | 118.2  | 149.6  | StackExchange post   | Web pages                                      |
| Robotics             | 101 | 62,198    | 5.5  | 120.6  | 818.9  | StackExchange post   | Web pages                                      |
| Stack Overflow       | 117 | 107,100   | 7.0  | 704.5  | 478.3  | StackExchange post   | Web pages                                      |
| Sustainable Living   | 108 | 60,732    | 5.6  | 108.0  | 148.5  | StackExchange post   | Web pages                                      |
| **Coding**                                                                                                                |
| LeetCode             | 142 | 413,932   | 1.8  | 483.1  | 497.5  | Coding question      | Coding Q&Sol                                   |
| Pony                 | 112 | 7,894     | 22.5 | 98.3   | 102.6  | Coding question      | Syntax Doc                                     |
| **Theorems**                                                                                                              |
| AoPS                 | 111 | 188,177   | 4.7  | 89.0   | 250.5  | Math Olympiad Q      | STEM Q&Sol                                     |
| TheoremQA            | 206 | 188,177   | 3.2  | 117.1  | 250.5  | Theorem-based Q      | STEM Q&Sol                                     |


---

# Retrieval Performance

## 1. SOTA performances

| Model     | Bio  | Earth | Econ | Psy  | Rob  | Stack | Sus  | Leet | Pony | AoPS | TheoQ | TheoT | Avg  |Checked |
|-----------|------|-------|------|------|------|--------|------|------|------|------|-------|--------|------|------|
| [Rank1-32B](https://arxiv.org/pdf/2502.18418) <br> (2025.02)   | 49.7 | 35.8  | 22.0 | 37.5 | 22.5 | 21.7  | 35.0 | 18.8 | 32.5  | 10.8  | 22.9  | 43.7   | 29.4 | :white_check_mark:     |
| [ReasonIR-8B](https://arxiv.org/pdf/2504.20595) <br> (2025.04) | 26.2 | 31.4  | 23.3 | 30.0 | 18.0 | 23.9  | 20.5 | 35.0 | 10.5  | 14.7  | 31.9  | 27.2   | 24.4 | :white_check_mark:     |


## 2. [Origial paper](https://arxiv.org/pdf/2407.12883), NDCG@10 performance

| Model     | Bio  | Earth | Econ | Psy  | Rob  | Stack | Sus  | Leet | Pony | AoPS | TheoQ | TheoT | Avg  | Checked |
|-----------|------|-------|------|------|------|--------|------|------|------|------|--------|--------|------|---------|
| BM25      | 18.9 | 27.2  | 14.9 | 12.5 | 13.6 | 18.4  | 15.0 | 24.4 | 7.9  | 6.2  | 10.4  | 4.9   | 14.5 | :white_check_mark:     |
| **Open-sourced models (<1B)** ||||||||||||||||
| BGE       | 11.7 | 24.6  | 16.6 | 17.5 | 11.7 | 10.8  | 13.3 | 26.7 | 5.7  | 6.0  | 13.0  | 6.9   | 13.7 | :white_check_mark:     |
| Inst-L    | 15.2 | 21.2  | 14.7 | 22.3 | 11.4 | 13.3  | 13.5 | 19.5 | 1.3  | 8.1  | 20.9  | 9.1   | 14.2 | :white_check_mark:     |
| SBERT     | 15.1 | 20.4  | 16.6 | 22.7 | 8.2  | 11.0  | 15.3 | 26.4 | 7.0  | 5.3  | 20.0  | 10.8  | 14.9 |  :white_check_mark:     |
| **Open-sourced models (>1B)** ||||||||||||||||
| E5        | 18.6 | 26.0  | 15.5 | 15.8 | 16.3 | 11.2  | 18.1 | 28.7 | 4.9  | 7.1  | 26.1  | 26.8  | 17.9 | :white_check_mark:      |
| SFR       | 19.1 | 26.7  | 17.8 | 19.0 | 16.3 | 14.4  | 19.2 | 27.4 | 2.0  | 7.4  | 24.3  | 26.0  | 18.3 |  :white_check_mark:     |
| Inst-XL   | 21.6 | 34.3  | 22.4 | 27.4 | 18.2 | 21.2  | 19.1 | 27.5 | 5.0  | 8.5  | 15.6  | 5.9   | 18.9 |  :white_check_mark:      |
| GritLM    | 24.8 | 32.3  | 18.9 | 19.8 | 17.1 | 13.6  | 17.8 | 29.9 | 22.0 | 8.8  | 25.2  | 21.2  | 21.0 |  :white_check_mark:     |
| Qwen      | **30.6** | **36.4**  | 17.8 | 24.6 | 13.2 | **22.2**  | 14.8 | 25.5 | 9.9  | **14.4** | **27.8**  | **32.9**  | **22.5** |  :white_check_mark:    |
| **Proprietary models** ||||||||||||||||
| Cohere    | 18.7 | 28.4  | 20.4 | 21.6 | 16.3 | 18.3  | 17.6 | 26.8 | 1.9  | 6.3  | 15.7  | 7.2   | 16.6 |  :white_check_mark:     |
| OpenAI    | 23.3 | 26.7  | 19.5 | 27.6 | 12.8 | 14.3  | 20.5 | 23.6 | 2.4  | 8.5  | 23.5  | 11.7  | 17.9 |  :white_check_mark:      |
| Voyage    | 23.1 | 25.4  | 19.9 | 24.9 | 10.8 | 16.5  | 15.4 | **30.6** | 1.5  | 7.5  | 27.4  | 11.6  | 17.9 |  :white_check_mark:   |
| Google    | 22.7 | 34.8  | 19.6 | **27.8** | 15.7 | 20.1  | 17.1 | 29.6 | 3.6  | 9.3  | 23.8  | 15.9  | 20.0 |  :white_check_mark:   |


---


