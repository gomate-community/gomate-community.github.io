const r=`from dataclasses import dataclass\r
from typing import List\r
import evaluate\r
\r
import datasets\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
\r
_DESCRIPTION = """\\\r
The AnswerAccuracy is to measure the correctness of answers.\r
\r
This metric is applicable in scenarios where the LLM is required to output a unique short answer, such as options for \\\r
multiple-choice questions or a single entity name.\r
The renowned MMLU dataset utilizes this metric for evaluation. In the evaluation of the MMLU dataset, probabilities \\\r
for each answer are first calculated, and the answer with the highest probability is selected as the predicted result.\r
In our tool, we assume that the prediction result has already been obtained, and only perform the final score \\\r
calculation.\r
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
    _compute_one: Evaluating the correctness of answer.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "A",\r
    ...         "B",\r
    ...         "C"\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         "A",\r
    ...         "C",\r
    ...         "C"\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerAccuracy()\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset["answers"], dataset["gt_answers"], 1)\r
    >>> score\r
    0.6666666666666666\r
    >>> results[0]\r
    True\r
"""\r
\r
_CITATION = """\\\r
@misc{hendrycks2021measuring,\r
      title={Measuring Massive Multitask Language Understanding},\r
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},\r
      year={2021},\r
      eprint={2009.03300},\r
      archivePrefix={arXiv},\r
      primaryClass={cs.CY}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerAccuracy(Metric):\r
    """Estimates the correctness of answers."""\r
\r
    name = "answer_accuracy"\r
\r
    ALIAS = ['answer_accuracy']\r
\r
    def __init__(self):\r
        """\r
        Explicitly initialize AnswerAccuracy.\r
\r
        Ensure all parent classes are initialized.\r
        """\r
        super().__init__()\r
        self.info = evaluate.MetricInfo(\r
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
            codebase_urls=["https://github.com/hendrycks/test"],\r
            reference_urls=["https://arxiv.org/abs/2009.03300"]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"\r
\r
    def _compute_one(\r
        self,\r
        answer: str,\r
        gt_answer: str\r
    ) -> float:\r
        """Evaluating the correctness of answer."""\r
        return answer == gt_answer\r
`;export{r as default};
