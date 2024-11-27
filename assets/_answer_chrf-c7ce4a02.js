const r=`from dataclasses import dataclass\r
from typing import List, Tuple, Optional\r
import evaluate\r
\r
import datasets\r
from sacrebleu.metrics import CHRF\r
import numpy as np\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
\r
_DESCRIPTION = """\\\r
    ChrF and ChrF++ are two MT evaluation metrics. They both use the F-score statistic for character n-gram matches, and ChrF++ adds word n-grams as well which correlates more strongly with direct assessment.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    predictions (list of str): The predicted sentences.\r
    references (list of list of str): The references. There should be one reference sub-list for each prediction sentence.\r
    char_order (int): Character n-gram order. Defaults to \`6\`.\r
    word_order (int): Word n-gram order. If equals to \`2\`, the metric is referred to as chrF++. Defaults to \`0\`.\r
    beta (int): Determine the importance of recall w.r.t precision. Defaults to \`2\`.\r
    lowercase (bool): if \`True\`, enables case-insensitivity. Defaults to \`False\`.\r
    whitespace (bool): If \`True\`, include whitespaces when extracting character n-grams.\r
    eps_smoothing (bool): If \`True\`, applies epsilon smoothing similar\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _validate_data: validate the dataset format.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "The relationship between cats and dogs is not exactly friendly.",\r
    ...         "a good bookshop is just a genteel black hole that knows how to read."\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         ["The relationship between dogs and cats is not exactly friendly.", ],\r
    ...         ["A good bookshop is just a genteel Black Hole that knows how to read."]\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerCHRFCorrectness()\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)\r
    >>> score\r
    84.64214891738334\r
    >>> results[0]\r
    84.41131092011067\r
"""\r
\r
_CITATION = """\\\r
@inproceedings{popovic-2015-chrf,\r
    title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",\r
    author = "Popovi{\\'c}, Maja",\r
    booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",\r
    month = sep,\r
    year = "2015",\r
    address = "Lisbon, Portugal",\r
    publisher = "Association for Computational Linguistics",\r
    url = "https://aclanthology.org/W15-3049",\r
    doi = "10.18653/v1/W15-3049",\r
    pages = "392--395",\r
}\r
@inproceedings{popovic-2017-chrf,\r
    title = "chr{F}++: words helping character n-grams",\r
    author = "Popovi{\\'c}, Maja",\r
    booktitle = "Proceedings of the Second Conference on Machine Translation",\r
    month = sep,\r
    year = "2017",\r
    address = "Copenhagen, Denmark",\r
    publisher = "Association for Computational Linguistics",\r
    url = "https://aclanthology.org/W17-4770",\r
    doi = "10.18653/v1/W17-4770",\r
    pages = "612--618",\r
}\r
@inproceedings{post-2018-call,\r
    title = "A Call for Clarity in Reporting {BLEU} Scores",\r
    author = "Post, Matt",\r
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",\r
    month = oct,\r
    year = "2018",\r
    address = "Belgium, Brussels",\r
    publisher = "Association for Computational Linguistics",\r
    url = "https://www.aclweb.org/anthology/W18-6319",\r
    pages = "186--191",\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerCHRFCorrectness(Metric):\r
    """Estimates the CHRF between answers and ground truth answers."""\r
\r
    name = "answer_chrf"\r
\r
    ALIAS = ['answer_chrf']\r
\r
    def __init__(\r
            self,\r
            char_order: int = 6,\r
            word_order: int = 0,\r
            beta: int = 2,\r
            lowercase: bool = False,\r
            whitespace: bool = False,\r
            eps_smoothing: bool = False\r
    ):\r
        """\r
        Explicitly initialize AnswerCHRFCorrectness.\r
\r
        Ensure all parent classes are initialized.\r
        """\r
        super().__init__()\r
\r
        self.chrf = CHRF(\r
            char_order=char_order,\r
            word_order=word_order,\r
            beta=beta,\r
            lowercase=lowercase,\r
            whitespace=whitespace,\r
            eps_smoothing=eps_smoothing\r
        )\r
        self.info = evaluate.MetricInfo(\r
            description=_DESCRIPTION,\r
            inputs_description=_KWARGS_DESCRIPTION,\r
            citation=_CITATION,\r
            features=datasets.Features(\r
                {\r
                    "answers": datasets.Value("string"),\r
                    "gt_answers": datasets.Sequence(datasets.Value("string"))\r
                }\r
            ),\r
            codebase_urls=["https://github.com/mjpost/sacreBLEU#chrf--chrf"],\r
            reference_urls=[\r
                "https://aclanthology.org/W15-3049.pdf",\r
                "https://aclanthology.org/W17-4770",\r
                "https://www.aclweb.org/anthology/W18-6319"\r
            ]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"  # pragma: no cover\r
\r
    def _validate_data(\r
        self,\r
        pred_answers: List[str],\r
        ref_answers: List[List[str]]\r
    ) -> None:\r
        """Validate the input dataset."""\r
        super()._validate_data(pred_answers, ref_answers)\r
        if not all(isinstance(answer, str) for answer in pred_answers):\r
            raise ValueError("The type of pred_answers should be a string.")  # pragma: no cover\r
        if not all(isinstance(a, list) and all(isinstance(item, str) for item in a) for a in ref_answers):\r
            raise ValueError("The type of ref_answers should be a list of strings.")  # pragma: no cover\r
\r
    def _compute_one(\r
        self,\r
        pred_answer: str,\r
        ref_answers: List[str]\r
    ) -> float:\r
        """Compute the metric for a single sentence against a single (or multiple) reference(s)."""\r
        return self.chrf.sentence_score(pred_answer, ref_answers).score\r
\r
    def compute(\r
        self,\r
        pred_answers: List[str],\r
        ref_answers: List[List[str]],\r
        batch_size: Optional[int] = None,\r
    ) -> Tuple[float, List[float]]:\r
        """Corpus score takes into account all the answers as two corpora and returns the F1 score of the corpus, which is not equal to the average of the chrF scores of the individual (pred, refs) pair."""\r
        self._validate_data(pred_answers, ref_answers)\r
        scores = self._compute_batch(pred_answers, ref_answers)\r
        ref_answers = np.array(ref_answers)\r
        ref_answers = ref_answers.T.tolist()\r
        return self.chrf.corpus_score(pred_answers, ref_answers).score, scores\r
`;export{r as default};
