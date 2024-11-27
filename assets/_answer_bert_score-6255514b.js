const e=`from dataclasses import dataclass\r
from typing import List, Tuple\r
import evaluate\r
\r
import datasets\r
from rageval.metrics import Metric, add_attribute\r
from bert_score import BERTScorer\r
import logging\r
import transformers\r
transformers.tokenization_utils.logger.setLevel(logging.ERROR)\r
transformers.configuration_utils.logger.setLevel(logging.ERROR)\r
transformers.modeling_utils.logger.setLevel(logging.ERROR)\r
\r
_DESCRIPTION = """\\\r
BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language generation tasks.\r
\r
For details, see the paper: https://openreview.net/forum?id=SkeHuCVFDr\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    lang : str, Language of the text. Default is "en".\r
    rescale_with_baseline : bool, Whether to rescale the score with pre-computed baseline. Not affect BERTScore's correlation with human judgment. Default is False. For more details, see https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _clean: clean special word in sentence.\r
    _compute_one: compute bleu score for single prediction with its references\r
    _compute_batch: compute bleu score for a batch of predictions with their references\r
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
    >>> metric = rl.metrics.AnswerBERTScore(lang='en', rescale_with_baseline=True)\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset["answers"], dataset["gt_answers"], 1)\r
    >>> round(score, 2)\r
    0.55\r
    >>> round(results[0], 1)\r
    0.7\r
"""\r
\r
\r
_CITATION = """\\\r
@inproceedings{bert-score,\r
    title={BERTScore: Evaluating Text Generation with BERT},\r
    author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},\r
    booktitle={International Conference on Learning Representations},\r
    year={2020},\r
    url={https://openreview.net/forum?id=SkeHuCVFDr}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerBERTScore(Metric):\r
    """BERTScore depends on the model and language pair selected."""\r
\r
    name = "answer_bert_score"\r
\r
    ALIAS = ['answer_bert_score']\r
\r
    def __init__(self, lang: str = "en", rescale_with_baseline=False):\r
        """Explicitly initialize the AnswerBERTScore to ensure all parent class initialized."""\r
        super().__init__()\r
        self.scorer = BERTScorer(lang=lang, rescale_with_baseline=rescale_with_baseline)\r
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
                "https://github.com/Tiiiger/bert_score/tree/master",\r
            ],\r
            reference_urls=["https://openreview.net/forum?id=SkeHuCVFDr"]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"\r
\r
    def _compute_one(\r
        self,\r
        pred_answers: str,\r
        ref_answers: List[str]\r
    ) -> float:\r
        """Compute the BERTscore for a pair of predictions and references."""\r
        P, R, F1 = self.scorer.score([pred_answers] * len(ref_answers), ref_answers)\r
        return F1.max().tolist()\r
`;export{e as default};
