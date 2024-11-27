const r=`import re\r
import string\r
from collections import Counter\r
from dataclasses import dataclass\r
from typing import List\r
import evaluate\r
\r
import datasets\r
import numpy as np\r
import spacy\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
_DESCRIPTION = """\\\r
The Disambig-F1 is a variant of the F1 score, estimates the similarity between the disambiguation of the answer and the ground truth answer.\r
\r
The original metric was presented in [ASQA paper](https://aclanthology.org/2022.emnlp-main.566/), and implemented through [this code](https://github.com/google-research/language/blob/master/language/asqa/scoring.py#L273). And we adopted an [alternative implementation](https://github.com/jzbjyb/FLARE/tree/main/src/datasets.py#L29) from the paper [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983).\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    model : str, model name of spacy model to ner.\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _normalize_text: normalize the text by removing articles, white spaces, punctuations and lowercasing.\r
    _ner: extract named entities from the text.\r
    _validate_data: validate the dataset format.\r
    _f1_score: compute the f1 score between \`pred\` string and \`ref\` string.\r
    _compute_one: evaluate the disambig f1 score of between \`answer\` and \`gt_answers\`, return the highest score in all pairs.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "Democrat rick kriseman won the 2016 mayoral election, while re- publican former mayor rick baker did so in the 2017 mayoral election."\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         [\r
    ...             "Kriseman",\r
    ...             "Rick Kriseman"\r
    ...         ]\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerDisambigF1Correctness(model="en_core_web_sm")\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)\r
    >>> assert 0 <= score <= 1\r
"""\r
\r
_CITATION = """\\\r
@inproceedings{stelmakh-etal-2022-asqa,\r
    title = "{ASQA}: Factoid Questions Meet Long-Form Answers",\r
    author = "Stelmakh, Ivan  and\r
      Luan, Yi  and\r
      Dhingra, Bhuwan  and\r
      Chang, Ming-Wei",\r
    editor = "Goldberg, Yoav  and\r
      Kozareva, Zornitsa  and\r
      Zhang, Yue",\r
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",\r
    month = dec,\r
    year = "2022",\r
    address = "Abu Dhabi, United Arab Emirates",\r
    publisher = "Association for Computational Linguistics",\r
    url = "https://aclanthology.org/2022.emnlp-main.566",\r
    doi = "10.18653/v1/2022.emnlp-main.566",\r
    pages = "8273--8288",\r
}\r
@misc{jiang2023active,\r
      title={Active Retrieval Augmented Generation},\r
      author={Zhengbao Jiang and Frank F. Xu and Luyu Gao and Zhiqing Sun and Qian Liu and Jane Dwivedi-Yu and Yiming Yang and Jamie Callan and Graham Neubig},\r
      year={2023},\r
      eprint={2305.06983},\r
      archivePrefix={arXiv},\r
      primaryClass={cs.CL}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerDisambigF1Correctness(Metric):\r
    """Estimates the Disambig-F1 between answers and ground truth answers."""\r
\r
    name = "answer_disambig_f1"\r
\r
    ALIAS = ['answer_disambig_f1']\r
\r
    def __init__(self, model: str = "en_core_web_sm"):\r
        """\r
        Explicitly initialize AnswerDisambigF1Correctness.\r
\r
        Ensure all parent classes are initialized.\r
        Ensure spacy ner model is initialized.\r
        """\r
        super().__init__()\r
        self.model = model\r
        self.nlp = spacy.load(model)\r
        self.info = evaluate.MetricInfo(\r
            description=_DESCRIPTION,\r
            inputs_description=_KWARGS_DESCRIPTION,\r
            citation=_CITATION,\r
            homepage="",\r
            features=datasets.Features(\r
                {\r
                    "answers": datasets.Value("string"),\r
                    "gt_answers": datasets.Sequence(datasets.Value("string"))\r
                }\r
            ),\r
            codebase_urls=[\r
                "https://github.com/google-research/language/blob/master/language/asqa",\r
                "https://github.com/jzbjyb/FLARE"\r
            ],\r
            reference_urls=[\r
                "https://aclanthology.org/2022.emnlp-main.566",\r
                "https://arxiv.org/abs/2305.06983"\r
            ]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"  # pragma: no cover\r
\r
    def _normalize_text(self, s: str) -> str:\r
        def remove_articles(text):\r
            return re.sub(r'\\b(a|an|the)\\b', ' ', text)\r
\r
        def white_space_fix(text):\r
            return ' '.join(text.split())\r
\r
        def remove_punc(text):\r
            exclude = set(string.punctuation)\r
            return ''.join(ch for ch in text if ch not in exclude)\r
\r
        def lower(text):\r
            return text.lower()\r
        return white_space_fix(remove_articles(remove_punc(lower(s))))\r
\r
    def _ner(self, s: str) -> List[str]:\r
        """Extract named entities from the text."""\r
        doc = self.nlp(s)\r
        ents = doc.ents\r
        return [self._normalize_text(e.text) for e in ents]\r
\r
    def _f1_score(self, pred: str, ref: str) -> float:\r
        """Compute the f1 score between pred and ref."""\r
        pred_ents = self._ner(pred)\r
        ref_ents = self._ner(ref)\r
\r
        pred_counter = Counter(pred_ents)\r
        ref_counter = Counter(ref_ents)\r
\r
        tp = sum((pred_counter & ref_counter).values())\r
        fp = sum((pred_counter - ref_counter).values())\r
        fn = sum((ref_counter - pred_counter).values())\r
\r
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1\r
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1\r
\r
        if precision + recall == 0:\r
            return 0\r
        return 2 * (precision * recall) / (precision + recall)\r
\r
    def _compute_one(\r
        self,\r
        pred_answer: str,\r
        ref_answers: List[str]\r
    ) -> float:\r
        """Evaluate the disambig f1 score of an answer."""\r
        return np.max([self._f1_score(pred_answer, ref_answer) for ref_answer in ref_answers])\r
`;export{r as default};
