const n=`import re\r
from dataclasses import dataclass\r
from typing import List, Callable\r
\r
import datasets\r
from tqdm import tqdm\r
\r
from rageval.metrics import Metric, add_attribute\r
from rageval.utils import text_to_sents, remove_citations\r
\r
_DESCRIPTION = """\\\r
Citation recall determines if the result from LLM is entirely supported by cited passages.\r
\r
In different RAG evaluation tasks, both ‘contexts’ and ‘gt_contexts’ may be used as part of the input of LLM.\r
This metric doesn't care whether the ‘contexts’ come from real-time retrieval or annotated datasets.\r
For simplicity, we refer to all contexts collectively as ‘contexts’.\r
\r
For details, see the paper: https://arxiv.org/abs/2305.14627.\r
"""\r
\r
_KWARGS_DESCRIPTION = r"""\\\r
Args:\r
    name : str\r
    batch_size : int, Batch size for openai completion.\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    _compute_one: compute the citation recall of an answer.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "answers": [\r
    ...         "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an "\r
    ...         "average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which "\r
    ...         "reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by "\r
    ...         "Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, "\r
    ...         "India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 "\r
    ...         "and most rain in a year from August 1860 to July 1861 [1]."\r
    ...     ],\r
    ...    "contexts": [\r
    ...        [\r
    ...             "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be "\r
    ...             "spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in "\r
    ...             "the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \\"hima\\" (Khasi tribal "\r
    ...             "chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often "\r
    ...             "been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds "\r
    ...             "that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar "\r
    ...             "month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received "\r
    ...             "in",\r
    ...             "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji "\r
    ...             "Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled "\r
    ...             "Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the "\r
    ...             "Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \\"hima\\" (Khasi tribal "\r
    ...             "chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often "\r
    ...             "been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds "\r
    ...             "that distinction. Cherrapunji still holds the all-time record for the most rainfall",\r
    ...             "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in "\r
    ...             "north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls "\r
    ...             "in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 "\r
    ...             "mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of "\r
    ...             "12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual "\r
    ...             "12,892 mm per year between 1960 and 2012. According to the \\"Guinness Book of World Records\\", "\r
    ...             "Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"\r
    ...        ]\r
    ...    ]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> nli_model = rl.models.NLIModel(\r
    ...     'text2text-generation',\r
    ...     'hf-internal-testing/tiny-random-T5ForConditionalGeneration'\r
    ... )\r
    >>> metric = rl.metrics.AnswerCitationRecall(nli_model=nli_model)\r
    >>> metric.mtype\r
    'AnswerGroundedness'\r
    >>> score, results = metric.compute(dataset['answers'], dataset['contexts'], 1)\r
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
@add_attribute('mtype', 'AnswerGroundedness')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class AnswerCitationRecall(Metric):\r
    """Estimates the citation recall of the generated answer based on the NLI model."""\r
\r
    name = "answer_citation_recall"\r
\r
    ALIAS = ['answer_citation_recall']\r
\r
    def __init__(self, nli_model: Callable):\r
        """\r
        Explicitly initialize AnswerCitationRecall.\r
\r
        Ensure all parent classes are initialized.\r
        Ensure nli_model is initialized.\r
        """\r
        super().__init__()\r
        self.nli_model = nli_model\r
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
            homepage="",\r
            features=datasets.Features(\r
                {\r
                    "answers": datasets.Value("string"),\r
                    "contexts": datasets.Sequence(datasets.Value("string"))\r
                }\r
            ),\r
            codebase_urls=["https://github.com/princeton-nlp/ALCE"],\r
            reference_urls=["https://arxiv.org/abs/2305.14627"]\r
        )\r
\r
    def _compute_one(\r
        self,\r
        answer: str,\r
        context: List[str]\r
    ) -> float:\r
        """Evaluate the citation recall of an answer."""\r
        total_entail = 0\r
\r
        sents = text_to_sents(answer)\r
        target_sents = [remove_citations(sent).strip() for sent in sents]\r
\r
        for idx, sent in enumerate(sents):\r
            target_sent = target_sents[idx]\r
\r
            context_ids = []\r
            for r in re.findall(r"\\[\\d+", sent):\r
                context_id = int(r[1:])\r
                if 1 <= context_id <= len(context):\r
                    context_ids.append(context_id)\r
                else:\r
                    context_ids = []\r
                    break\r
\r
            if len(context_ids) > 0:\r
                # citation id starts from 1 in sents\r
                premise = " ".join([context[context_id - 1] for context_id in context_ids])\r
                label = self.nli_model.generate_infer(premise=premise, hypothesis=target_sent)\r
                total_entail += label\r
\r
        if len(sents) == 0:\r
            return 0\r
        return total_entail / len(sents)\r
\r
    def _compute_batch(\r
        self,\r
        answers: List[str],\r
        contexts: List[List[str]]\r
    ) -> List[float]:\r
        """\r
        Evaluate the citation recall of a batch of answers.\r
\r
        Firstly, calculate the citation recall of each statement (0 or 1).\r
        For each statement, its citation recall is 1 if and only if there is at least one citation and connects all\r
        paragraphs cited by this statement as premise, statement as hypothesis, and the NLI model outputs 1\r
        when it determines that premise entails the hypothesis, otherwise it is 0.\r
        Then, average over all statements in the LLM answer.\r
        Finally, average over all scores of each answer.\r
        """\r
        return super()._compute_batch(pred_answers=answers, ref_answers=contexts)\r
`;export{n as default};
