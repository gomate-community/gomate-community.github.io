const r=`# -*- coding: utf-8 -*-\r
\r
from dataclasses import dataclass\r
from typing import List, Callable, Optional\r
\r
import datasets\r
from datasets import Dataset\r
from rouge_score import rouge_scorer\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
_DESCRIPTION = """Estimates ROUGE score by estimating answer and groundtruth answers.\r
\r
ROUGE is case insensitive, so the input text is converted to lower case before computing the score. This metrics is a wrapper around the https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py\r
\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    rouge_type : str, the rouge type to calculate. Defaults to 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'\r
        "rouge1": unigram (1-gram) based scoring\r
        "rouge2": bigram (2-gram) based scoring\r
        "rougeL": Longest common subsequence based scoring.\r
        "rougeLSum": splits text using "\\n".\r
\r
Optional Args:\r
    tokenizer : Callable, a tokenizer can be passed to the scorer, replacing the default tokenizer which tokenizes on whitespace, especially for non-latin languages. For example, the \`jieba.cut\` can be used for Chinese.\r
\r
Functions:\r
    _compute_one: compute the score by measure whether the args:\`answer\` contains short answer in list:\`gt_answers\`.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...    "answers": [\r
    ...        "Some nanomaterials may give rise to various kinds of lung damage."\r
    ...    ],\r
    ...    "gt_answers":[\r
    ...        [\r
    ...            "Nanoparticles can penetrate the body, affecting the lungs, brain, and other organs,\\\r
    ...             leading to possible respiratory, cardiovascular, and brain health problems.",\r
    ...            "Due to their small size, nanoparticles can infiltrate the body and impact vital organs,\\\r
    ...             posing risks to respiratory, heart, and neurological health."\r
    ...        ]\r
    ...    ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerRougeCorrectness('rougeL')\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)\r
    >>> assert 0 <= score <= 1\r
"""\r
\r
_CITATION = """\\\r
@inproceedings{lin-2004-rouge,\r
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",\r
    author = "Lin, Chin-Yew",\r
    booktitle = "Text Summarization Branches Out",\r
    month = jul,\r
    year = "2004",\r
    address = "Barcelona, Spain",\r
    publisher = "Association for Computational Linguistics",\r
    url = "https://aclanthology.org/W04-1013",\r
    pages = "74--81",\r
}\r
@article{lewis2020retrieval,\r
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},\r
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\\"a}schel, Tim and others},\r
  journal={Advances in Neural Information Processing Systems},\r
  volume={33},\r
  pages={9459--9474},\r
  year={2020}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerRougeCorrectness(Metric):\r
\r
    name = "answer_rouge_correctness"\r
\r
    ALIAS = ['answer_rouge_correctness']\r
\r
    def __init__(self, rouge_type: str, tokenizer: Optional[Callable] = None):\r
        """Explicitly initialize the AnswerRougeCorrectness to ensure all parent class initialized as well as initialize the rouge type and tokenizer."""\r
        self.rouge_type = rouge_type\r
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True, tokenizer=tokenizer)\r
        super().__init__()\r
\r
    def __repr__(self) -> str:\r
        """:return: Formated string representation of the metric."""\r
        return f"{self.ALIAS[0]}"\r
\r
    def _info(self):\r
        return datasets.MetricInfo(\r
            description=_DESCRIPTION,\r
            inputs_description=_KWARGS_DESCRIPTION,\r
            citation=_CITATION,\r
            homepage="",\r
            features=datasets.Features(\r
                {\r
                    "answers": datasets.Value("string", id="sequence"),\r
                    "gt_answers": datasets.Value("string", id="sequence"),\r
                }\r
            ),\r
            codebase_urls=["https://github.com/mim-solutions/rouge_score"],\r
            reference_urls=[\r
                "https://aclanthology.org/W04-1013/",\r
                "https://arxiv.org/abs/2005.11401"\r
            ]\r
        )\r
\r
    def _compute_one(self, pred_answer: str, ref_answers: List[str]) -> float:\r
        """Evaluate the ROUGE between a single answer and groundtruth answers."""\r
        score = self.scorer.score_multi(ref_answers, pred_answer)\r
        return score[self.rouge_type].fmeasure\r
`;export{r as default};
