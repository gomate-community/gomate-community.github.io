const r=`from dataclasses import dataclass\r
from typing import List\r
\r
import datasets\r
import numpy as np\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
\r
_DESCRIPTION = """\\\r
AnswerEMCorrectness evaluates answer correctness based on exact matching of annotated short answers.\r
\r
For details, see the paper: https://arxiv.org/abs/2204.06092.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    batch_size : int, Batch size for openai completion.\r
    ignore_case : bool, whether to ignore case when comparing the answer and ground truth answers.\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _compute_one: compute the score by measure whether the args:\`answer\` contains short answer in list:\`gt_answers\`.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "Ali Dael has the highest goals in men's world international football with 109 goals. Josef Bican has "\r
    ...         "the highest goals all-time in men's football and Christine Sinclair has the highest goals in women's "\r
    ...         "world international football.",\r
    ...         "A supercentenarian is someone who has reached the age of 110. Sarah Knauss, whose age is undisputed, "\r
    ...         "was the oldest person ever from the United States and the second-oldest fully documented person ever. "\r
    ...         "Jeanne Calment was a French supercentenarian and the oldest human whose age is well-documented, with "\r
    ...         "a lifespan of 122 years and 164 days, and was the oldest person in the world as of 1997. In 1985, "\r
    ...         "the oldest living person was Mathew Beard and in 1986 it was Augusta Holtz, who lived 115 years and "\r
    ...         "79 days, from 1871 to 1986."\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         [\r
    ...             ["Daei", "Ali Daei"],\r
    ...             ["Bican", "Josef Bican"],\r
    ...             ["Sinclair","Christine Sinclair"]\r
    ...         ],\r
    ...         [\r
    ...             ["Jeanne Calment"],\r
    ...             ["Sarah Knauss"],\r
    ...             ["Augusta-Holtz"],\r
    ...         ]\r
    ...     ],\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerEMCorrectness()\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)\r
    >>> assert 0 <= score <= 1\r
"""\r
\r
_CITATION = """\\\r
@misc{stelmakh2023asqa,\r
      title={ASQA: Factoid Questions Meet Long-Form Answers},\r
      author={Ivan Stelmakh and Yi Luan and Bhuwan Dhingra and Ming-Wei Chang},\r
      year={2023},\r
      eprint={2204.06092},\r
      archivePrefix={arXiv},\r
      primaryClass={cs.CL}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerEMCorrectness(Metric):\r
    """Estimates correctness using annotated short answers."""\r
\r
    name = "answer_exact_match"\r
\r
    ALIAS = ['answer_exact_match']\r
\r
    def __init__(self, ignore_case: bool = False):\r
        """Explicitly initialize the AnswerEMCorrectness to ensure all parent class initialized."""\r
        super().__init__()\r
        self.ignore_case = ignore_case\r
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
                    "gt_answers": datasets.Sequence(datasets.Value("string"))\r
                }\r
            ),\r
            codebase_urls=[],\r
            reference_urls=["https://arxiv.org/abs/2204.06092"]\r
        )\r
\r
    def _compute_one(self, pred_answer: str, short_answers: List[List[str]]) -> float:\r
        """Compute the correctness of a single answer."""\r
        acc = []\r
        if self.ignore_case:\r
            pred_answer = pred_answer.lower()\r
            short_answers = [[a.lower() for a in candidate_short_answers] for candidate_short_answers in short_answers]\r
        for candidate_short_answers in short_answers:\r
            for candidate_short_answer in candidate_short_answers:\r
                if candidate_short_answer in pred_answer:\r
                    acc.append(True)\r
                    break\r
            else:\r
                acc.append(False)\r
        return np.average(acc)\r
`;export{r as default};
