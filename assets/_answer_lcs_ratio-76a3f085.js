const r=`from dataclasses import dataclass\r
from typing import List\r
import datasets\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
_DESCRIPTION = """\\\r
The AnswerLCSRatio is to measure the similarity between answer and gt_answer by calculating the longest common subsequence.\r
\r
This is a very traditional method, but to this day, some work is still being carried out using it, such as \\\r
https://ieeexplore.ieee.org/abstract/document/10172590.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    batch_size : int, Batch size for openai completion.\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _compute_one: evaluating the similarity between answer and gt_answer by calculating the longest common subsequence.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "Language models trained on massive code corpora can generalize to tasks without the need "\r
    ...         "for task-specific fine-tuning."\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         "Large language models trained on massive code corpora can generalize to new tasks without the need "\r
    ...         "for task-specific fine-tuning."\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerLCSRatio()\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)\r
    >>> assert score == 16 / 17\r
"""\r
\r
_CITATION = """\\\r
@INPROCEEDINGS{10172590,\r
    author={Nashid, Noor and Sintaha, Mifta and Mesbah, Ali},\r
    booktitle={2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)},\r
    title={Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning},\r
    year={2023},\r
    volume={},\r
    number={},\r
    pages={2450-2462},\r
    doi={10.1109/ICSE48619.2023.00205}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerLCSRatio(Metric):\r
    """Estimates the similarity between answers and gt_answers."""\r
\r
    name = "answer_lcs_ratio"\r
\r
    ALIAS = ['answer_lcs_ratio']\r
\r
    def __init__(self):\r
        """\r
        Explicitly initialize AnswerLCSRatio.\r
\r
        Ensure all parent classes are initialized.\r
        """\r
        super().__init__()\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
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
                    "answers": datasets.Value("string"),\r
                    "gt_answers": datasets.Value("string")\r
                }\r
            ),\r
            codebase_urls=[],\r
            reference_urls=["https://ieeexplore.ieee.org/abstract/document/10172590"]\r
        )\r
\r
    def _compute_one(\r
        self,\r
        pred_answer: str,\r
        ref_answer: str\r
    ) -> float:\r
        """Evaluating the similarity between answer and gt_answer by calculating the longest common subsequence."""\r
        pred_answer = pred_answer.split()\r
        ref_answer = ref_answer.split()\r
        m, n = len(pred_answer), len(ref_answer)\r
\r
        if m == 0 or n == 0:\r
            return 0\r
\r
        dp = [0] * (n + 1)\r
        for i in range(m):\r
            pre = 0\r
            for j in range(n):\r
                tmp = dp[j + 1]\r
                dp[j + 1] = pre + 1 if pred_answer[i] == ref_answer[j] else max(dp[j + 1], dp[j])\r
                pre = tmp\r
\r
        return dp[-1] / m\r
`;export{r as default};
