const r=`from dataclasses import dataclass\r
from typing import List, Tuple\r
\r
import datasets\r
from sacrebleu.metrics import TER\r
import numpy as np\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
_DESCRIPTION = """\\\r
    TER (Translation Edit Rate, also called Translation Error Rate) is a metric to quantify the edit operations that a\r
hypothesis requires to match a reference translation. The implementation is already present in sacrebleu\r
(https://github.com/mjpost/sacreBLEU#ter), which in turn is inspired by the TERCOM implementation, which can be found\r
here: https://github.com/jhclark/tercom.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    normalized (boolean): If \`True\`, applies basic tokenization and normalization to sentences. Defaults to \`False\`.\r
    ignore_punct (boolean): If \`True\`, applies basic tokenization and normalization to sentences. Defaults to \`False\`.\r
    support_zh_ja_chars (boolean): If \`True\`, tokenization/normalization supports processing of Chinese characters,\r
                                    as well as Japanese Kanji, Hiragana, Katakana, and Phonetic Extensions of Katakana.\r
                                    Only applies if \`normalized = True\`. Defaults to \`False\`.\r
    case_sensitive (boolean): If \`False\`, makes all predictions and references lowercase to ignore differences in case. Defaults to \`False\`.\r
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
    ...         "does this sentence match??",\r
    ...         "what about this sentence?",\r
    ...         "What did the TER metric user say to the developer?"\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         ["does this sentence match", "does this sentence match!?!"],\r
    ...         ["wHaT aBoUt ThIs SeNtEnCe?", "wHaT aBoUt ThIs SeNtEnCe?"],\r
    ...         ["Your jokes are...", "...TERrible"]\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerTERCorrectness()\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'])\r
    >>> assert score == 110.00000000000001\r
    >>> assert results[0] == 25.0\r
"""\r
\r
_CITATION = """\\\r
@inproceedings{snover-etal-2006-study,\r
    title = "A Study of Translation Edit Rate with Targeted Human Annotation",\r
    author = "Snover, Matthew  and\r
      Dorr, Bonnie  and\r
      Schwartz, Rich  and\r
      Micciulla, Linnea  and\r
      Makhoul, John",\r
    booktitle = "Proceedings of the 7th Conference of the Association for Machine Translation in the Americas: Technical Papers",\r
    month = aug # " 8-12",\r
    year = "2006",\r
    address = "Cambridge, Massachusetts, USA",\r
    publisher = "Association for Machine Translation in the Americas",\r
    url = "https://aclanthology.org/2006.amta-papers.25",\r
    pages = "223--231",\r
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
class AnswerTERCorrectness(Metric):\r
    """Estimates the TER between answers and ground truth answers."""\r
\r
    name = "answer_ter"\r
\r
    ALIAS = ['answer_ter']\r
\r
    def __init__(\r
        self,\r
        normalized: bool = False,\r
        ignore_punct: bool = False,\r
        support_zh_ja_chars: bool = False,\r
        case_sensitive: bool = False\r
    ):\r
        """\r
        Explicitly initialize AnswerTERCorrectness.\r
\r
        Ensure all parent classes are initialized.\r
        """\r
        super().__init__()\r
        self.ter = TER(\r
            normalized=normalized,\r
            no_punct=ignore_punct,\r
            asian_support=support_zh_ja_chars,\r
            case_sensitive=case_sensitive\r
        )\r
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
                    "answers": datasets.Value("string"),\r
                    "gt_answers": datasets.Sequence(datasets.Value("string"))\r
                }\r
            ),\r
            codebase_urls=["https://github.com/mjpost/sacreBLEU#ter"],\r
            reference_urls=["https://aclanthology.org/2006.amta-papers.25", "https://www.aclweb.org/anthology/W18-6319"]\r
        )\r
\r
    def _validate_data(\r
        self,\r
        pred_answers: List[str],\r
        ref_answers: List[List[str]]\r
    ) -> None:\r
        """Validate the input predictions and references."""\r
        super()._validate_data(pred_answers, ref_answers)\r
        if not all(isinstance(pred_answer, str) for pred_answer in pred_answers):\r
            raise ValueError("The type of pred_answers should be a list of strings.")\r
        if not all(isinstance(reference_list, list) and all(isinstance(reference, str) for reference in reference_list) for reference_list in ref_answers):\r
            raise ValueError("The type of ref_answers should be a list of lists of strings.")\r
\r
    def _compute_one(\r
        self,\r
        pred_answer: str,\r
        ref_answers: List[str]\r
    ) -> float:\r
        """Compute the TER score of a single answer."""\r
        return self.ter.sentence_score(pred_answer, ref_answers).score\r
\r
    def compute(\r
        self,\r
        pred_answers: List[str],\r
        ref_answers: List[List[str]],\r
    ) -> Tuple[float, List[float]]:\r
        """Evaluate the dataset."""\r
        self._validate_data(pred_answers, ref_answers)\r
        scores = self._compute_batch(pred_answers, ref_answers)\r
        ref_answers = np.array(ref_answers)\r
        ref_answers = ref_answers.T.tolist()\r
        return self.ter.corpus_score(pred_answers, ref_answers).score, scores\r
`;export{r as default};
