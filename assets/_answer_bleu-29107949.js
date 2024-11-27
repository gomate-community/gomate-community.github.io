const e=`import re\r
from dataclasses import dataclass\r
from typing import List, Tuple\r
import evaluate\r
import datasets\r
from rageval.metrics import Metric, add_attribute\r
from tqdm import tqdm\r
\r
\r
_DESCRIPTION = """\\\r
BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.\r
Scores are calculated by comparing individual translated segments, e.g., sentences, with a set of high-quality reference translations.\r
Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality.\r
Neither intelligibility nor grammatical correctness are not taken into account.\r
\r
For details, see the paper: http://www.aclweb.org/anthology/P02-1040.pdf\r
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
    _clean: clean special word in sentence.\r
    _compute_one: compute bleu score for single prediction with its references\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "It is a guide to action which ensures that the military always obeys the commands of the party.",\r
    ...         "It is to insure the troops forever hearing the activity guidebook that party direct.",\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         [\r
    ...             "It is a guide to action that ensures that the military will forever heed Party commands.",\r
    ...             "It is the guiding principle which guarantees the military forces always being under the command of the Party.",\r
    ...             "It is the practical guide for the army always to heed the directions of the party.",\r
    ...         ],\r
    ...         [\r
    ...             "It is a guide to action that ensures that the military will forever heed Party commands.",\r
    ...             "It is the guiding principle which guarantees the military forces always being under the command of the Party.",\r
    ...             "It is the practical guide for the army always to heed the directions of the party.",\r
    ...         ]\r
    ...     ],\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerBleuScore()\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset["answers"], dataset["gt_answers"], 1)\r
    >>> score\r
    0.3450835085970013\r
    >>> results[0]\r
    0.5401725898595141\r
"""\r
\r
\r
_CITATION = """\\\r
@misc{Kishore2002bleu,\r
      title={Bleu: a method for automatic evaluation of machine translation},\r
      author={Kishore Papineni and Salim Roukos and Todd Ward and Wei-Jing Zhu},\r
      year={2002},\r
      page={311-318},\r
      primaryClass={cs.CL}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerBleuScore(Metric):\r
    """Bleu score computing with good quality reference."""\r
\r
    """Note: this metric is just fit for English data by now(24/03/12)"""\r
\r
    name = "answer_bleu"\r
\r
    ALIAS = ['answer_bleu']\r
\r
    def __init__(self):\r
        """Explicitly initialize the AnswerBleuScore to ensure all parent class initialized."""\r
        super().__init__()\r
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
                "https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py",\r
                "https://github.com/huggingface/datasets/blob/main/metrics/bleu/bleu.py"\r
            ],\r
            reference_urls=["https://www.aclweb.org/anthology/P02-1040.pdf"]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"  # pragma: no cover\r
\r
    def compute(\r
        self,\r
        pred_answers: List[str],\r
        ref_answers: List[List[str]],\r
        batch_size: int,\r
    ) -> Tuple[float, List[float]]:\r
        """Compute the bleu score on both corpus level and instance level."""\r
        bleu = evaluate.load("bleu")\r
        # corpus level\r
        bleu_result = bleu.compute(predictions=pred_answers, references=ref_answers)\r
        score = bleu_result['bleu']\r
        # instance level\r
        scores = []\r
        for pred_answer, ref_answer in tqdm(zip(pred_answers, ref_answers),\r
                                            desc=f"Computing {self.name}",\r
                                            total=len(pred_answers)):\r
            scores.append(self._compute_one(pred_answer, ref_answer))\r
        return score, scores\r
\r
    def _compute_one(\r
        self,\r
        pred_answers: List[str],\r
        ref_answers: List[List[str]]\r
    ) -> List[float]:\r
        """Compute the bleu score on an instance level."""\r
\r
        bleu = evaluate.load("bleu")\r
        bleu_result = bleu.compute(predictions=[pred_answers], references=[ref_answers])\r
        bleu_score = bleu_result['bleu']\r
        return bleu_score\r
`;export{e as default};
