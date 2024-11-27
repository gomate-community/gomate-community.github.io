const r=`import re\r
import string\r
from collections import Counter\r
from dataclasses import dataclass\r
from typing import List, Optional, Iterable, Union\r
\r
import datasets\r
import numpy as np\r
import jieba\r
\r
from rageval.metrics import Metric, add_attribute\r
\r
\r
_DESCRIPTION = """\\\r
    F1 score combines precision and recall into a single score using their harmonic mean.\r
"""\r
\r
_KWARGS_DESCRIPTION = """\\\r
Args:\r
    name : str\r
    normalize : bool, default is True, whether to normalize the text. If False, the text will be treated as a list of tokens.\r
    language : str, default is 'en', the language of the text. Supported languages are 'en' and 'zh'.\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _normalize_text: normalize the text by removing articles, white spaces, punctuations and lowercasing.\r
    _validate_data: validate the dataset format.\r
    _f1_score: compute the f1 score between \`pred\` tokens and \`ref\` tokens.\r
    _compute_one: evaluate the f1 score of between \`answer\` and \`gt_answers\`, return the highest score in all pairs.\r
\r
Examples:\r
    English:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "Democrat rick kriseman won the 2016 mayoral election, while re- publican former mayor rick baker did so in the 2017 mayoral election."\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         [\r
    ...             "Kriseman",\r
    ...             "Rick Kriseman"\r
    ...         ]\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerF1Correctness()\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'])\r
    >>> round(score, 2)\r
    0.18\r
\r
    Chinese:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "督邮，中国古代职官名，自汉代开始设置。",\r
    ...         "魏晋",\r
    ...         "北齐只设于清都郡。",\r
    ...         "隋代",\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         ["督邮，中国古代职官名，自汉代开始设置。"],\r
    ...         ["魏晋", "魏晋时期"],\r
    ...         ["北齐只设于清都郡。", "清都郡"],\r
    ...         ["隋代", "隋朝"]\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerF1Correctness(language='zh')\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'])\r
    >>> round(score, 2)\r
    1.0\r
\r
    Other Iterables:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [[1,2,3], [4,5,6]],\r
    ...     "gt_answers": [[2,3,4,5,6], [1,2,3,4,5]]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> metric = rl.metrics.AnswerF1Correctness(normalize=False)\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'])\r
    >>> round(score, 2)\r
    0.5\r
\r
"""\r
\r
_CITATION = """\\\r
\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerF1Correctness(Metric):\r
    """Estimates the F1 between answers and ground truth answers."""\r
\r
    name = "answer_f1"\r
\r
    ALIAS = ['answer_f1']\r
\r
    def __init__(self, normalize: bool = True, language: Optional[str] = "en"):\r
        """\r
        Explicitly initialize AnswerF1Correctness.\r
\r
        Ensure all parent classes are initialized.\r
        """\r
        super().__init__()\r
        self.normalize = normalize\r
        self.language = language\r
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
            codebase_urls=[],\r
            reference_urls=[]\r
        )\r
\r
    def _normalize_text(self, s: str) -> List[str]:\r
        def remove_articles(text):\r
            return re.sub(r'\\b(a|an|the)\\b', ' ', text)\r
\r
        def remove_punc(text):\r
            exclude = set(string.punctuation)\r
            return ''.join(ch for ch in text if ch not in exclude)\r
\r
        def lower(text):\r
            return text.lower()\r
        return remove_articles(remove_punc(lower(s))).split()\r
\r
    def _normalize_text_zh(self, s: str) -> str:\r
        """Normalize Chinese text."""\r
        def white_space_fix(text):\r
            return ' '.join(text.split())\r
\r
        def remove_punc(text):\r
            exclude = set(string.punctuation) | {'，', '。', '？', '！', '：', '；', '“', '”', '‘', '’', '（', '）', '《', '》', '——', '……', '、'}\r
            return ''.join(ch for ch in text if ch not in exclude)\r
\r
        return white_space_fix(remove_punc(s))\r
\r
    def _f1_score(self, pred: Iterable, ref: Iterable) -> float:\r
        """Compute the f1 score between pred and ref."""\r
        pred_counter = Counter(pred)\r
        ref_counter = Counter(ref)\r
\r
        tp = sum((pred_counter & ref_counter).values())\r
        fp = sum((pred_counter - ref_counter).values())\r
        fn = sum((ref_counter - pred_counter).values())\r
\r
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1\r
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1\r
\r
        if precision + recall == 0:\r
            return 0\r
        return 2 * (precision * recall) / (precision + recall)\r
\r
    def _compute_one(\r
        self,\r
        pred_answer: Union[str, Iterable],\r
        ref_answers: Union[List[str], Iterable]\r
    ) -> float:\r
        """Evaluate the f1 score of an answer."""\r
        if self.normalize:\r
            # str, List[str] -> List[str], List[List[str]]\r
            if self.language == "en":\r
                preds = self._normalize_text(pred_answer)\r
                refs = [self._normalize_text(ref_answer) for ref_answer in ref_answers]\r
            elif self.language == "zh":\r
                preds = list(jieba.cut(self._normalize_text_zh(pred_answer)))\r
                refs = [list(jieba.cut(self._normalize_text_zh(ref_answer))) for ref_answer in ref_answers]\r
            else:\r
                raise Exception('Unsupported language: {}'.format(self.language))  # pragma: no cover\r
            scores = [self._f1_score(preds, ref) for ref in refs]\r
        else:\r
            scores = self._f1_score(pred_answer, ref_answers)\r
\r
        return np.max(scores)\r
`;export{r as default};
