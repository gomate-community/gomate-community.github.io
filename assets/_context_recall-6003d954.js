const r=`from dataclasses import dataclass\r
from typing import Callable, List, Tuple\r
import evaluate\r
\r
import datasets\r
import numpy as np\r
import pandas as pd\r
from langchain.schema import LLMResult\r
from tqdm import tqdm\r
\r
from rageval.metrics import MetricWithLLM, add_attribute\r
from rageval.utils.utility import json_loader\r
from rageval.utils import CONTEXT_RECALL_RA\r
\r
_DESCRIPTION = """\\\r
ContextRecall evaluates contexts relevancy based on gt_answers.\r
\r
For details, see the doc: https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html.\r
"""\r
\r
_KWARGS_DESCRIPTION = r"""\\\r
Args:\r
    name : str\r
    batch_size : int, Batch size for openai completion.\r
    model : Callable, The LLM model to use.\r
\r
Optional Args:\r
    None\r
\r
Functions:\r
    parse_llm_result: Parse the LLM Result based on the Prompt\r
    _compute_batch: Compute the score by measure whether the args:\`contexts\` can be supported by args:\`gt_answers\`.\r
\r
Examples:\r
    >>> from datasets import Dataset\r
    >>> from langchain.llms.fake import FakeListLLM\r
    >>> import rageval as rl\r
    >>> sample = {\r
    ...     "questions": ["恐龙是怎么被命名的？"],\r
    ...     "gt_answers": [["1841年，英国科学家理查德·欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙，意思是“恐怖的蜥蜴”。"]],\r
    ...     "contexts": [["[12]恐龙是 介于冷血和温血之间的动物2014年6月，有关恐龙究竟是像鸟类和哺乳动物一样的温血动物，还是类似爬行动物、鱼类和两栖动物的冷血动物的问题终于有了答案——恐龙其实是介于冷血"\r
    ...                   "和温血之间的动物。 [12]“我们的结果显示恐龙所具有的生长速率和新陈代谢速率，既不是冷血生物体也不是温血生物体所具有的特征。它们既不像哺乳动物或者鸟类，也不像爬行动物或者鱼类，"\r
    ...                   "而是介于现代冷血动物和温血动物之间。简言之，它们的生理机能在现代社会并不常见。”美国亚利桑那大学进化生物学家和生态学家布莱恩·恩奎斯特说。墨西哥生物学家表示，正是这种中等程度的"\r
    ...                   "新陈代谢使得恐龙可以长得比任何哺乳动物都要大。温血动物需要大量进食，因此它们频繁猎捕和咀嚼植物。“很难想象霸王龙大小的狮子能够吃饱以 存活下来。","[12]哺乳动物起源于爬行动物，"\r
    ...                   "它们的前身是“似哺乳类的爬行动物”，即兽孔目，早期则是“似爬行类的哺乳动物”，即哺乳型动物。 [12]中生代的爬行动物，大部分在中生代的末期灭绝了；一部分适应了变化的环境被保留下来，"\r
    ...                   "即现存的爬行动物（如龟鳖类、蛇类、鳄类等）；还有一部分沿着不同的进化方向，进化成了现今的鸟类和哺乳类。 [12]恐龙是 介于冷血和温血之间的动物2014年6月，有关恐龙究竟是像鸟类和"\r
    ...                   "哺乳动物一样的温血动物，还是类似爬行动物、鱼类和两栖动物的冷血动物的问题终于有了答案——恐龙其实是介于冷血和温血之间的动物。"\r
    ...                 ]]\r
    ... }\r
    >>> dataset = Dataset.from_dict(sample)\r
    >>> model = FakeListLLM(\r
    ...     responses=['[\\n    {\\n        "statement_1":"恐龙的命名始于1841年，由英国科学家理查德·欧文命名。",\\n        "reason": "The answer '\r
    ...                'provides the exact year and the scientist who named the dinosaurs.",\\n        "Attributed": "1"'\r
    ...                '\\n    },\\n    {\\n        "statement_2":"欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙。",'\r
    ...                '\\n        "reason": "The answer accurately describes the process of how dinosaurs were named.",'\r
    ...                '\\n        "Attributed": "1"\\n    }\\n]'\r
    ...               ]\r
    ... )\r
    >>> metric = rl.metrics.ContextRecall(model)\r
    >>> metric.mtype\r
    'ContextRelevancy'\r
    >>> score, results = metric.compute(dataset['questions'], dataset['gt_answers'], dataset['contexts'], 1)\r
    >>> assert 0 <= score <= 1\r
"""\r
\r
_CITATION = """\r
@misc{ragas,\r
    author= {explodinggradients},\r
    year  = {2023},\r
    title = {ragas},\r
    note  = {https://github.com/explodinggradients/ragas, Last accessed on 2024-3-2},\r
}\r
"""\r
\r
\r
@dataclass\r
@add_attribute('mtype', 'ContextRelevancy')\r
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)\r
class ContextRecall(MetricWithLLM):\r
    """Estimates context recall by estimating TP and FN using annotated answer and retrieved context."""\r
\r
    name = "context_recall"\r
\r
    ALIAS = ['context_recall']\r
\r
    def __init__(self, model: Callable):\r
        """Explicitly initialize the AnswerEMCorrectness to ensure all parent class initialized."""\r
        super().__init__(model)\r
        self.info = evaluate.MetricInfo(\r
            description=_DESCRIPTION,\r
            inputs_description=_KWARGS_DESCRIPTION,\r
            citation=_CITATION,\r
            homepage="",\r
            features=datasets.Features(\r
                {\r
                    "questions": datasets.Value("string"),\r
                    "gt_answers": datasets.Sequence(datasets.Value("string")),\r
                    "contexts": datasets.Sequence(datasets.Value("string"))\r
                }\r
            ),\r
            codebase_urls=["https://github.com/explodinggradients/ragas"],\r
            reference_urls=["https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html"]\r
        )\r
\r
    def __repr__(self) -> str:\r
        """:return: Formatted string representation of the metric."""\r
        return f"{self.ALIAS[0]}"  # pragma: no cover\r
\r
    def parse_llm_result(self, prompts: str, result: LLMResult):\r
        """\r
        Parse the LLM Result based on the Prompt.\r
\r
        TODO: use prompts to parse the result.\r
        """\r
        results = []\r
        scores = []\r
        responses = [[i.text for i in r] for r in result.generations]\r
        # for each question-answer pair\r
        for response in responses:\r
            response = json_loader.safe_load(response[0], self.llm)\r
            # response: list of dict; each dict is a statement extracted from gt_answer\r
            if response:\r
                reasonings = [\r
                    str(item)\r
                    for item in response\r
                ]\r
                score = [\r
                    int(item.get("Attributed", "0").strip() == "1")\r
                    if item.get("Attributed")\r
                    else np.nan\r
                    for item in response\r
                ]\r
                data = {'reasoning': reasonings, 'score': score}\r
                scores.append(np.average(score))\r
            else:\r
                data = {'reasoning': [np.nan], 'score': [0.]}\r
                scores.append(0.)\r
            results.append(pd.DataFrame(data))\r
        # Note that the \`results can be recorded by logger.info\`\r
        return scores\r
\r
    def compute(\r
        self,\r
        questions: List[str],\r
        ref_answers: List[str],\r
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
                ref_answers[start:end],\r
                contexts[start:end]\r
            )\r
            scores.extend(score)\r
        return np.average(scores), scores\r
\r
    def _compute_batch(\r
        self,\r
        questions: List[str],\r
        ref_answers: List[str],\r
        contexts: List[List[str]]\r
    ) -> List[float]:\r
\r
        prompts = []\r
        for question, ref_answer, context in zip(questions, ref_answers, contexts):\r
            ref_answer = "\\n".join(ref_answer) if isinstance(ref_answer, list) else ref_answer\r
            context = "\\n".join(context) if isinstance(context, list) else context\r
            prompt = CONTEXT_RECALL_RA.format(\r
                question=question,\r
                context=context,\r
                answer=ref_answer\r
            )\r
            prompts.append(prompt)\r
\r
        result = self.llm.generate(prompts)\r
        scores = self.parse_llm_result(prompts, result)\r
\r
        return scores\r
`;export{r as default};
