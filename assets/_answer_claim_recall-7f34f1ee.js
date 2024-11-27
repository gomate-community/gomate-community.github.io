const e=`from dataclasses import dataclass\r
from typing import List, Callable, Tuple\r
import evaluate\r
\r
import datasets\r
import numpy as np\r
from tqdm import tqdm\r
\r
from rageval.metrics import Metric, add_attribute\r
from rageval.utils.check_utils import text_to_sents\r
\r
_DESCRIPTION = """\\\r
The AnswerNLICorrectness is to measure the correctness of long-form answers. In the original paper, the author first \\\r
use Instruct-GPT(text-davinci-003) to generate three "sub-claims" (based on gold answers) and use a state-of-the-art \\\r
natural-language inference (NLI) model TRUE(Honovich et al., 2022) to check whether the model output entails the \\\r
sub-claims (claim recall).\r
\r
For details, see the paper: https://arxiv.org/abs/2305.14627.\r
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
    _verify_by_stance: verify whether the stance of args:\`claims\` can be supported by args:\`answer\`.\r
    _compute_one: compute the score by measure whether the args:\`claims\` can be supported by args:\`answers\`.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "They went a while before introducing ads, so they could make money, as they needed to  establish "\r
    ...         "their brand and amass users. Once you have dedicated users, introducing ads won't deter most, but if "\r
    ...         "you are still new, having ads will deter a lot. The same goes for Uber, it's not that they aren't "\r
    ...         "making money, it's that they are reinvesting a ton of it to make their service better."\r
    ...     ],\r
    ...     "gt_answers": [\r
    ...         [\r
    ...             "Firms like Snapchat and Uber need to establish their brand and amass users before introducing "\r
    ...             "ads.",\r
    ...             "Introducing ads too early can deter potential users.",\r
    ...             "Uber is reinvesting a lot of money to make their service better."\r
    ...         ]\r
    ...     ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> nli_model = rl.models.NLIModel(\r
    ...     'text2text-generation',\r
    ...     'hf-internal-testing/tiny-random-T5ForConditionalGeneration'\r
    ... )\r
    >>> metric = rl.metrics.AnswerNLICorrectness(nli_model=nli_model, decompose_model="nltk")\r
    >>> metric.mtype\r
    'AnswerCorrectness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['gt_answers'], 1)\r
    >>> assert score == 0 or score == 1\r
"""\r
\r
_CITATION = """\\\r
@misc{gao2023enabling,\r
      title={Enabling Large Language Models to Generate Text with Citations},\r
      author={Tianyu Gao and Howard Yen and Jiatong Yu and Danqi Chen},\r
      year={2023},\r
      eprint={2305.14627},\r
      archivePrefix={arXiv},\r
      primaryClass={cs.CL}\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'AnswerCorrectness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerNLICorrectness(Metric):\r
    """Estimates the correctness of long-form answers based on the NLI model."""\r
\r
    name = "answer_claim_recall"\r
\r
    ALIAS = ['answer_claim_recall']\r
\r
    def __init__(self, nli_model: Callable, decompose_model: str = "gpt-3.5-turbo"):\r
        """\r
        Explicitly initialize AnswerNLICorrectness.\r
\r
        Ensure all parent classes are initialized.\r
        Ensure nli_model and decompose_model is initialized.\r
        """\r
        super().__init__()\r
        self.nli_model = nli_model\r
        self.decompose_model = decompose_model\r
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
            codebase_urls=["https://github.com/princeton-nlp/ALCE"],\r
            reference_urls=["https://arxiv.org/abs/2305.14627"]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"  # pragma: no cover\r
\r
    def _compute_one(\r
        self,\r
        answer: str,\r
        claims: List[str]\r
    ) -> float:\r
        """\r
        Evaluate the correctness of an answer.\r
\r
        Firstly, split the gt_answer into a set of claims.\r
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.\r
        Finally, aggregate all faithfulness score of each claim.\r
        """\r
\r
        detail_results = []\r
        scores = []\r
\r
        for i, claim in enumerate(claims):\r
            # obtain the faithfulness of each claim by language inference model.\r
            label = self.nli_model.generate_infer(premise=answer, hypothesis=claim)\r
            detail_results.append({\r
                "answer": answer,\r
                "claim": claim,\r
                "reasoning": "",\r
                "error": "",\r
                "factuality": label,\r
            })\r
            scores.append(label)\r
        # Note that the detail_results can be recorded by logger.info\r
        return np.average(scores)\r
\r
    def _compute_batch(\r
        self,\r
        pred_answers: List[str],\r
        ref_answers: List[List[str]]\r
    ) -> List[float]:\r
        """\r
        Evaluate the correctness of a batch of answers.\r
\r
        Firstly, split the gt_answer into a set of claims.\r
        Then, compute the faithfulness score of each claim. The faithfulness is a binary score.\r
        Finally, aggregate all faithfulness score of each claim.\r
        """\r
\r
        if isinstance(ref_answers, list):\r
            if isinstance(ref_answers[0], list):\r
                # gt_answers has been decomposed into claims list\r
                claims = ref_answers\r
            elif isinstance(ref_answers[0], str):\r
                # use decompose_model to decompose the gt_answers into claims list\r
                claims = [text_to_sents(gt_answer, self.decompose_model) for gt_answer in ref_answers]\r
            else:\r
                raise ValueError("The type of gt_answers element should be list or string.")  # pragma: no cover\r
        else:\r
            raise ValueError("The type of gt_answers should be list.")  # pragma: no cover\r
\r
        results = []\r
        for i, answer in tqdm(enumerate(pred_answers)):\r
            r = self._compute_one(answer, claims[i])\r
            results.append(r)\r
        return results\r
`;export{e as default};
