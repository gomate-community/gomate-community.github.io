const r=`from typing import List, Tuple, Callable, Optional, Iterable\r
from abc import abstractmethod\r
from dataclasses import dataclass\r
\r
import numpy as np\r
from langchain.schema import LLMResult\r
from tqdm import tqdm\r
\r
\r
def add_attribute(attribute_name, attribute_value):\r
    """\r
    This decorate is used to set attribute for Class.\r
\r
    Currently, this decorate can be used to set attr:metric_type for each metric.\r
    There are four types, i.e., 'AnswerCorrectness', 'AnswerGroundedness', 'ContextRelevancy', 'ContextAdequacy', \\\r
    for all RAG metrics.\r
    """\r
    def decorator(cls):\r
        setattr(cls, attribute_name, attribute_value)\r
        return cls\r
    return decorator\r
\r
\r
@dataclass\r
class Metric():\r
    """Metric base class without LLM."""\r
\r
    def __init__(\r
        self,\r
        config_name: Optional[str] = None,\r
        experiment_id: Optional[str] = None\r
    ):\r
        """Initialization.\r
\r
        Args:\r
            config_name: type(string), Optional.\r
            experiment_id: type(string), Optional.\r
        """  # pragma: no cover\r
\r
    @property\r
    @abstractmethod\r
    def name(self) -> str:\r
        """The metric name."""\r
        ...  # pragma: no cover\r
\r
    def _validate_data(\r
        self,\r
        pred_answers: Optional[Iterable] = None,\r
        ref_answers: Optional[Iterable] = None,\r
        *args: Optional[Iterable]\r
    ) -> None:\r
        """Validate the of the input dataset."""\r
        if (pred_answers and ref_answers):\r
            if len(pred_answers) != len(ref_answers) or any(len(pred_answers) != len(arg) for arg in args):\r
                raise ValueError("The length of predictions and references should be the same.")  # pragma: no cover\r
\r
    def compute(\r
        self,\r
        pred_answers: Optional[Iterable] = None,\r
        ref_answers: Optional[Iterable] = None,\r
        batch_size: Optional[int] = None,\r
        *args: Optional[Iterable],\r
    ) -> Tuple[float, List[float]]:\r
        """\r
        Evaluate the dataset.\r
\r
        Return average scores of all inputs and a score list for each example.\r
        """\r
        self._validate_data(pred_answers, ref_answers, *args)\r
        scores = self._compute_batch(pred_answers, ref_answers, *args)\r
\r
        return np.average(scores), scores\r
\r
    @abstractmethod\r
    def _compute_one(\r
        self,\r
        pred_answer: Optional[Iterable] = None,\r
        ref_answer: Optional[Iterable] = None,\r
        *args: Optional[Iterable]\r
    ) -> float:\r
        ...  # pragma: no cover\r
\r
    def _compute_batch(\r
        self,\r
        pred_answers: Optional[Iterable] = None,\r
        ref_answers: Optional[Iterable] = None,\r
        *args: Optional[Iterable]\r
    ) -> List[float]:\r
        """Compute the metric for a batch of predictions and references."""\r
        scores = []\r
        if (pred_answers and ref_answers):  # if both columns exist\r
            for pred_answer, ref_answer in tqdm(zip(pred_answers, ref_answers),\r
                                                desc=f"Computing {self.name}",\r
                                                total=len(pred_answers)):\r
                scores.append(self._compute_one(pred_answer, ref_answer))\r
        else:\r
            for pred_answer in tqdm(pred_answers,\r
                                    desc=f"Computing {self.name}",\r
                                    total=len(pred_answers)):\r
                scores.append(self._compute_one(pred_answer))\r
        return scores\r
\r
\r
@dataclass\r
class MetricWithLLM(Metric):\r
    """Metrics based on LLM."""\r
\r
    def __init__(self, model: Callable):\r
        """Initialization."""\r
        super().__init__()\r
        self.llm = model\r
\r
    @abstractmethod\r
    def parse_llm_result(self, prompts: List[str], result: LLMResult):\r
        """Parse the LLM Result based on the Prompt."""\r
        ...  # pragma: no cover\r
`;export{r as default};
