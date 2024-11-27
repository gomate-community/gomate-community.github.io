const e=`from dataclasses import dataclass\r
from typing import Callable, List, Tuple\r
\r
import datasets\r
import numpy as np\r
import evaluate\r
\r
from langchain.schema import LLMResult\r
from tqdm import tqdm\r
\r
from rageval.metrics import MetricWithLLM, add_attribute\r
from rageval.utils.prompt import REJECT_RATE_PROMPT\r
\r
_DESCRIPTION = """\\\r
ContextRejectRate is the metric to measure the unknown robustness of LLM based on the given context.\r
\r
For details, see the paper: https://arxiv.org/abs/2311.09210.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    batch_size : int, Batch size for openai completion.\r
    model : Callable, The LLM model to use.\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    parse_llm_result: parse the results of LLM\r
    _compute_batch: compute the score by measure how many rejected answers in all answers.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> from langchain.llms.fake import FakeListLLM\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "questions": [\r
    ...         "Why did Bushnell set himself on fire?",\r
    ...         "Did Bushnell have a wife?"\r
    ...     ],\r
    ...     "contexts": [\r
    ...         [\r
    ...             ["An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "\r
    ...              "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "\r
    ...              "genocide.”"],\r
    ...             ["The 25-year-old airman, Aaron Bushnell, of San Antonio, Texas, died from his injuries, the "\r
    ...              "Metropolitan Police Department said Monday."],\r
    ...             ["Bushnell had walked up to the embassy shortly before 1 p.m. Sunday and began livestreaming on "\r
    ...              "the video streaming platform Twitch, a person familiar with the matter told The Associated "\r
    ...              "Press. Law enforcement officials believe he set his phone down and then doused himself in "\r
    ...              "accelerant and ignited the flames. At one point, he said he “will no longer be complicit in "\r
    ...              "genocide,” the person said. The video was later removed from the platform, but law enforcement "\r
    ...              "officials have obtained and reviewed a copy."]\r
    ...         ],\r
    ...         [\r
    ...             ["An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "\r
    ...              "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "\r
    ...              "genocide.”"],\r
    ...             ["The 25-year-old airman, Aaron Bushnell, of San Antonio, Texas, died from his injuries, the "\r
    ...              "Metropolitan Police Department said Monday."],\r
    ...             ["Bushnell had walked up to the embassy shortly before 1 p.m. Sunday and began livestreaming on "\r
    ...              "the video streaming platform Twitch, a person familiar with the matter told The Associated "\r
    ...              "Press. Law enforcement officials believe he set his phone down and then doused himself in "\r
    ...              "accelerant and ignited the flames. At one point, he said he “will no longer be complicit in "\r
    ...              "genocide,” the person said. The video was later removed from the platform, but law enforcement "\r
    ...              "officials have obtained and reviewed a copy."]\r
    ...         ],\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> model = FakeListLLM(\r
    ...     responses=[\r
    ...         "Answer: An active-duty member of the U.S. Air Force has died after he set himself ablaze outside the "\r
    ...         "Israeli Embassy in Washington, D.C., while declaring that he “will no longer be complicit in "\r
    ...         "genocide.”",\r
    ...         "Answer: sorry, cannot answer the question"\r
    ...     ]\r
    ... )\r
    >>> metric = rl.metrics.ContextRejectRate(model)\r
    >>> metric.mtype\r
    'AnswerGroundedness'\r
    >>> score, results = metric.compute(dataset['questions'], dataset['contexts'], 1)\r
    >>> assert 0 <= score <= 1\r
"""\r
\r
_CITATION = """\\\r
@misc{yu2023chainofnote,\r
      title={Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models},\r
      author={Wenhao Yu and Hongming Zhang and Xiaoman Pan and Kaixin Ma and Hongwei Wang and Dong Yu},\r
      year={2023},\r
      eprint={2311.09210},\r
      archivePrefix={arXiv},\r
      primaryClass={cs.CL}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerGroundedness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class ContextRejectRate(MetricWithLLM):\r
    """Estimates context reject rate by measuring how many rejected answers in all answers."""\r
\r
    name = "context_reject_rate"\r
\r
    ALIAS = ['context_reject_rate']\r
\r
    def __init__(self, model: Callable):\r
        """Explicitly initialize the ContextRejectRate to ensure all parent class initialized."""\r
        super().__init__(model)\r
        self.info = evaluate.MetricInfo(\r
            description=_DESCRIPTION,\r
            inputs_description=_KWARGS_DESCRIPTION,\r
            citation=_CITATION,\r
            homepage="",\r
            features=datasets.Features(\r
                {\r
                    "questions": datasets.Value("string"),\r
                    "contexts": datasets.Sequence(datasets.Value("string"))\r
                }\r
            ),\r
            codebase_urls=[],\r
            reference_urls=["https://arxiv.org/abs/2311.09210"]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"  # pragma: no cover\r
\r
    def parse_llm_result(self, prompts: List[str], result: LLMResult):\r
        """Parse the results of LLM based on whether the answer contains the content specified by prompt."""\r
        responses = [[i.text for i in r] for r in result.generations]\r
        scores = []\r
        # for each question-answer pair\r
        for response in responses:\r
            answer = response[0]\r
            if "sorry, cannot answer the question" in answer:\r
                scores.append(1.)\r
            else:\r
                scores.append(0.)\r
        return scores\r
\r
    def compute(\r
        self,\r
        questions: List[str],\r
        contexts: List[List[str]],\r
        batch_size: int,\r
    ) -> Tuple[float, List[float]]:\r
        """Evaluate the dataset."""\r
        scores = []\r
        length = len(questions)\r
        for start in tqdm(range(0, length, batch_size)):\r
            end = start + batch_size\r
            end = end if end < length else length\r
            score = self._compute_batch(\r
                questions[start:end],\r
                contexts[start:end]\r
            )\r
            scores.extend(score)\r
\r
        return np.average(scores), scores\r
\r
    def _compute_batch(\r
        self,\r
        questions: List[str],\r
        contexts: List[List[str]],\r
    ) -> List[float]:\r
        """Compute the score by measure how many rejected answers in all answers."""\r
\r
        prompts = []\r
        for question, context in zip(questions, contexts):\r
            prompt = REJECT_RATE_PROMPT.format(\r
                question=question,\r
                evidence=context\r
            )\r
            prompts.append(prompt)\r
\r
        results = self.llm.generate(prompts)\r
        scores = self.parse_llm_result(prompts, results)\r
        return scores\r
`;export{e as default};
