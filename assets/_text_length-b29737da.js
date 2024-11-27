const e=`from dataclasses import dataclass\r
from typing import Optional, Iterable\r
from transformers import AutoTokenizer\r
import evaluate\r
\r
import datasets\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
\r
_DESCRIPTION = """\\\r
Textlength is a metric used to evaluate the length of a model-generated response.\r
\r
It measures the number of tokens in the generated text by first converting the text into tokens and then counting the total number. This metric provides insight into the verbosity or conciseness of the model's output, offering a standardized way to compare text length across different responses.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _compute_one: Evaluating the length of answer.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "A",\r
    ...         "C",\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = TextLength(tokenize_model="Qwen/Qwen2-0.5B-Instruct")\r
    >>> metric.mtype\r
    'answer_informativeness'\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'answer_informativeness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class TextLength(Metric):\r
    """Estimates the text length of answers."""\r
\r
    name = "text_length"\r
\r
    ALIAS = ['text_length']\r
\r
    def __init__(self, tokenize_model: str = "Qwen/Qwen2-0.5B-Instruct"):\r
        """\r
        Explicitly initialize TextLength.\r
\r
        Ensure all parent classes are initialized.\r
        """\r
        self.tokenizer = AutoTokenizer.from_pretrained(tokenize_model)\r
        super().__init__()\r
        self.info = evaluate.MetricInfo(\r
            description=_DESCRIPTION,\r
            inputs_description=_KWARGS_DESCRIPTION,\r
            citation="",\r
            homepage="",\r
            features=datasets.Features(\r
                {\r
                    "answers": datasets.Value("string"),\r
                }\r
            ),\r
            codebase_urls=[],\r
            reference_urls=[]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"  # pragma: no cover\r
\r
    def _compute_one(\r
        self,\r
        answer: str,\r
        *args: Optional[Iterable],\r
    ) -> float:\r
        """Evaluating the text length of answer."""\r
        length = len(self.tokenizer(answer, return_tensors="pt")['input_ids'][0])\r
        return length\r
`;export{e as default};
