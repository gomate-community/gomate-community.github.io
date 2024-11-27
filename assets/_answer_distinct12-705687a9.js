const r=`from collections import Counter\r
from dataclasses import dataclass\r
from typing import List, Optional, Iterable, Tuple\r
import datasets\r
from nltk import ngrams\r
from rageval.metrics import Metric, add_attribute\r
\r
_DESCRIPTION = """\\\r
Distinct 1/2 measures the diversity of generated text by calculating the ratio of unique n-grams to the total number of n-grams.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    pred_answers (list of str): List of generated texts for which distinct metrics are computed.\r
    n_grams (int): The n-gram order for which distinct metrics are computed.\r
\r
Returns:\r
    dict: Dictionary containing Distinct-1 and Distinct-2 scores.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "This is the first sentence.",\r
    ...         "This is the second sentence."\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerDistinct(1)\r
    >>> metric.mtype\r
    'AnswerInformativeness'\r
    >>> score, results = metric.compute(dataset['answers'])\r
    >>> score\r
    0.6\r
"""\r
\r
_CITATION = """\\\r
@misc{selfmemory2023,\r
    title={Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory},\r
    author={Xin Cheng and Di Luo and Xiuying Chen and Lemao Liu and Dongyan Zhao and Rui Yan},\r
    year={2023},\r
    eprint={2305.02437},\r
    archivePrefix={arXiv},\r
    primaryClass={cs.CL}\r
}\r
"""\r
\r
\r
def get_distinct_score(pred_answers: List[str], n_grams: int) -> dict:\r
    """Compute Distinct-1 and Distinct-2 metrics."""\r
    c = Counter()\r
    for answer in pred_answers:\r
        tokens = answer.split()\r
        c.update(ngrams(tokens, n_grams))\r
\r
    return len(c) / sum(c.values())\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerInformativeness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerDistinct(Metric):\r
    """Distinct 1/2 metric for text generation."""\r
\r
    name = "answer_distinct"\r
\r
    ALIAS = ['answer_distinct']\r
\r
    def __init__(self, n_grams: int = 1):\r
        """\r
        Explicitly initialize Distinct.\r
\r
        Ensure all parent classes are initialized.\r
        """\r
        super().__init__()\r
        self.n_grams = n_grams\r
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
            features=datasets.Features(\r
                {\r
                    "pred_answers": datasets.Value("string"),\r
                }\r
            ),\r
            codebase_urls=["https://github.com/Hannibal046/SelfMemory/blob/main/src/utils/metrics_utils.py"],\r
            reference_urls=["https://arxiv.org/abs/2305.02437"]\r
        )\r
\r
    def _validate_data(\r
        self,\r
        pred_answers: Optional[Iterable] = None,\r
        ref_answers: Optional[Iterable] = None,\r
    ) -> bool:\r
        """Validate the input data."""\r
        assert isinstance(pred_answers, str) or isinstance(pred_answers, list)  # pragma: no cover\r
\r
    def compute(\r
        self,\r
        pred_answers: Optional[Iterable] = None,\r
    ) -> Tuple[float, List[float]]:\r
        """\r
        Evaluate the dataset.\r
\r
        Return average scores of all inputs and a score list for each example.\r
        """\r
        return get_distinct_score(pred_answers, self.n_grams), [get_distinct_score([pred_answer], self.n_grams) for pred_answer in pred_answers]\r
`;export{r as default};
