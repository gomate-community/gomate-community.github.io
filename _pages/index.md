---
layout: splash
permalink: /
title: "Gomate Community"
header:
  overlay_color: "#666"
  overlay_filter: "0.3"
  # overlay_image: /assets/images/header.jpg
  actions:
    - label: "GitHub"
      url: "https://github.com/gomate-community"
excerpt: "Projects and papers from Gomate Community, focused on RAG research, IR benchmarks, and trustworthy LLMs."
---

<style>
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem; /* 保持卡片之间的间距 */
  margin-top: 2rem;
}
.archive__item {
  border: 1px solid #eee;
  border-radius: 10px;
  overflow: hidden;
  background: #fff;
  transition: transform 0.2s, box-shadow 0.2s;
  /* 移除 height: 100%; 如果存在于此或内联样式中 */
}
.archive__item:hover {
  transform: translateY(-5px);
  box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
}
.card-grid a {
  display: block;
  text-decoration: none;
  color: inherit;
  height: 100%; /* 确保链接占据整个卡片高度，如果卡片高度不同时有助于点击区域 */
}

/* 图片容器 */
.archive__item-image-container {
  height: 100px; /* 进一步减小图片容器高度，从 100px 减到 90px */
  overflow: hidden;
  /* 确保这里没有额外的 padding 或 margin */
}

/* 图片本身 */
.archive__item-image-container img {
  width: 100%;
  height: 100%; /* 确保图片完全填充容器 */
  object-fit: cover;
  display: block; /* 移除图片底部可能存在的额外空白 */
}

/* 文本内容区域 */
.archive__item-content {
  padding: 0em 1em 0.1em; /* 进一步减小上下内边距：上0em，下0.1em，左右1em */
  display: flex;
  flex-direction: column;
  justify-content: space-between; /* 保持内容垂直方向上的分布 */
}

.archive__item-title {
  margin-top: 0;
  margin-bottom: 0 em; /* 进一步减小标题底部边距 */
  font-size: 1.1em; /* 稍微减小字体大小，使其更紧凑 */
  line-height: 1.1; /* 进一步减小行高 */
  color: #333;
  word-break: break-word;
}

.archive__item-excerpt {
  margin-bottom: 0em; /* 进一步减小描述底部边距 */
  font-size: 0.85em; /* 稍微减小字体大小 */
  line-height: 1.1; /* 调整行高 */
  color: #555;
  display: -webkit-box;
  -webkit-line-clamp: 2; /* 保持最多显示2行 */
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-grow: 1; /* 让描述区域尽可能占据空间 */
}

.archive__item-stats { /* 星标和数字区域 */
  margin-top: 0em; /* 进一步减小顶部边距 */
  margin-bottom: 0; /* 移除底部边距 */
  font-size: 0.4em; /* 进一步减小字体大小 */
  color: #555;
  text-align: left;
}
</style>

<div class="card-grid">

{% include feature_card.html
    title="TrustRAG"
    excerpt="The RAG Framework within Reliable Input, Trusted Output."
    url="/trustrag"
    icon="fab fa-python"
    repo="TrustRAG"
    image="/assets/images/trustrag-cover.png"
%}

{% include feature_card.html
    title="rageval"
    excerpt="Evaluation tools for Retrieval-Augmented Generation (RAG) methods."
    url="/rageval"
    icon="fab fa-python"
    repo="rageval"
    image="/assets/images/rageval-cover.png"
%}

{% include feature_card.html
    title="awesome-papers-for-rag"
    excerpt="A curated list of RAG-related papers and resources."
    url="/papers"
    icon="fab fa-python"
    repo="awesome-papers-for-rag"
    image="/assets/images/awesome-papers-for-rag-cover.png"
%}

{% include feature_card.html
    title="BenchRAG"
    excerpt="Benchmarking datasets used in RAG evaluation."
    url="/benchrag"
    icon="fas fa-database"
    repo="BenchRAG"
    image="/assets/images/benchrag-cover.png"
%}

{% include feature_card.html
    title="Must-Read-IR-Papers"
    excerpt="Best, test-of-time and highly cited IR papers."
    url="https://github.com/gomate-community/Must-Read-IR-Papers"
    icon="fas fa-book"
    repo="Must-Read-IR-Papers"
    image="/assets/images/must-read-ir-papers-cover.png"
%}


</div>

<script>
fetch('/assets/data/repos.json')
  .then(res => res.json())
  .then(repos => {
    repos.forEach(repo => {
      const el = document.getElementById(`stats-${repo.name}`);
      if (el) {
        el.innerHTML = `⭐ Starred ${repo.stargazers_count} &nbsp;&nbsp;| 🍴 Fork ${repo.forks_count}`;
      }
    });
  });
</script>

