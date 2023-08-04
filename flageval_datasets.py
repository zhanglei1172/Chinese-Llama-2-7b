import copy
import json
import random

import datasets
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_pt_utils import LabelSmoother


def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)


def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

huggingface_datasets = ["RAFT", "TruthfulQA", "IMDB", "BoolQ", "MMLU"]
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CEvalDataset(Dataset):
    dummy_message = {
            "system": "这个任务是中国关于civil考试的问题，请从给出的A、B、C、D四个选项中，选出其中的正确答案。请回答'A'或'B'或'C'或'D'\n",
            "conversations": [
                {"from": "human","value":"这个任务是中国关于civil考试的问题，请从给出的A、B、C、D四个选项中，选出其中的正确答案。请回答'A'或'B'或'C'或'D'\n\n\n请效仿此示例：问题:1， 2， 2， 4， ____， 32\n选项：A. 6\nB. 8\nC. 16\nD. 24\n答案: B\n\n请效仿此示例：问题:浪漫的时代总富于瑰丽的想象，灾难的岁月自然免不了灰暗的色彩，普罗米修斯的千秋功过就这样交替在“恩人”与“罪人”这两极间频繁地晃动着，让人____。难怪在学养深厚的经典注疏家维斯特看来，研究文献虽汗牛充栋，其实却____。 填入画横线部分最恰当的一项是____。\n选项：A. 捉摸不定 平淡无奇\nB. 无所适从 乏善可陈\nC. 扑朔迷离 差强人意\nD. 眼花缭乱 不赞一词\n答案: C\n\n请效仿此示例：问题:一个世界范围的对生产某些破坏臭氧层的化学物质的禁令只能提供一种受到保护的幻觉。已经生产出的大量的这种化学物质已经作为制冷剂存在于数百万台冰箱中。一旦它们到达大气中的臭氧层，它们引起的反应无法被停止。因此没有办法来阻止这些化学物质进一步破坏臭氧层。下面哪项最能加强上述的论述?____\n选项：A. 人们无法准确测出作为冰箱制冷剂存在的破坏臭氧层的化学物质的数量\nB. 在现代社会，为避免不健康甚至对生命构成潜在威胁的状况，冷藏食物是必要的\nC. 即使人们放弃使用冰箱，早已存在于冰箱中的制冷剂还是会威胁大气中的臭氧\nD. 冰箱中的制冷剂可以在冰箱完成使命后被完全开发并重新使用\n答案: C\n\n请效仿此示例：问题:军队的战斗力取决于武器装备和人员素质。在2008年与俄罗斯的军队冲突中损失惨重的格鲁吉亚，准备花费90亿美元，用现代化装备重新武装自己的军队。尽管美国非常支持格鲁吉亚加强军事力量，却不准备将先进的武器卖给它。以下各项陈述，除哪项陈述外，都可以解释美国的这种做法?____\n选项：A. 俄罗斯准备要求安理会对格鲁吉亚实行武器禁运\nB. 格鲁吉亚军队为这场战争准备了3年，尽管全副美式装备，却不堪一击\nC. 格军的战机在开战后数小时就放弃起飞，巡逻艇直接被俄军俘获并用卡车运走\nD. 格军的一名高级将领临阵脱逃，把部队丢弃不顾\n答案: A\n\n请效仿此示例：问题:下列情形哪一项属于自首?____\n选项：A. 甲杀人后其父主动报案并将甲送到派出所，甲当即交代了杀人的全部事实和经过\nB. 甲和乙共同贪污之后，主动到检察机关交代自己的贪污事实，但未提及乙\nC. 甲和乙共同盗窃之后，主动向公安机关反映乙曾经诈骗数千元，经查证属实\nD. 甲给监察局打电话，承认自己收受他人1万元贿赂，并交代了事情经过，然后出走不知所踪\n答案: B\n\n问题:1，0，9，16，____，48\n选项：\nA. 33\nB. 25\nC. 36\nD. 42\n答案: "}, {"from":"gpt", "value":"B"}],
            
        }
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, ceval_path,using_gpt=False, item_size=5):
        super().__init__()
        self.tokenizer = tokenizer
        with open(ceval_path, "r", encoding="utf-8") as file:
            self.dataset = json.load(file)
        if len(ceval_path.split("\\")) > 1:
            subject_name = " ".join(ceval_path.split("\\")[-1][:-4].split("_")[:-1])
        else:
            subject_name = " ".join(ceval_path.split("/")[-1][:-4].split("_")[:-1])
        self.name = "CEval/" + subject_name
        # self.name=subject_name
        self.first_line = (
            "这个任务是中国关于"
            + subject_name
            + "考试的问题，请从给出的A、B、C、D四个选项中，选出其中的正确答案。请回答'A'或'B'或'C'或'D'\n"
        )
        if using_gpt:  # 是否使用gpt的提示作为first line
            prompts = [
                "请回答下列问题。",
                "在每个问题中选择正确的答案。",
                "请仔细阅读问题并选择最合适的选项。",
                "在每个问题中选择最恰当的答案。",
                "请从给定的选项中选择正确的答案以回答问题。",
                "根据问题要求选择最佳答案。",
                "在每个问题的选项中，找出与问题描述最匹配的答案。",
                "请根据问题描述，从提供的选项中选择正确的答案。",
                "在每个问题中，选择最适合的选项以回答问题。",
                "根据问题描述，从给定的选项中选择最合适的答案。",
                "请从提供的选项中选择最适合问题描述的答案。",
                "根据问题描述，选择下列选项中最准确的答案。",
                "在每个问题中，从给定的选项中选择最符合要求的答案。",
                "请仔细阅读每个问题，并从给定的选项中选择最适合的答案。",
                "根据问题描述，在每个问题中选择最佳答案以回答问题。",
            ]
            idx = random.sample(range(0, 15), 1)[0]
            self.first_line = prompts[idx]
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        json_data = self.dataset
        idns = random.sample(range(0, len(json_data)), self.item_size)
        max_try = 10
        while ban_index in idns and max_try > 0:
            idns = random.sample(range(0, len(json_data)), self.item_size)
            max_try -= 1
        if max_try == 0:
            print("Warning: cannot generate prompt without Question index")
        prompt = self.first_line
        for idx in idns:
            entry = json_data[idx]
            question = entry["question"]
            choices = entry["choices"]
            answer = entry["answer"]
            # answer = entry["answer"]+'.'+choices[ord(entry["answer"])-65]

            formatted_string = f"请效仿此示例：问题:{question}\n"
            formatted_string += "选项："
            formatted_string += "\n".join(
                [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
            )
            formatted_string += f"\n答案: {answer}"

            prompt = prompt + "\n\n" + formatted_string
        return prompt

    def __getitem__(self, index):
        # prompt = self.first_line
        # if torch.is_tensor(index):
        #     index = index.tolist()
        # if index is iterable:
        # if not isinstance(index, list):
        #     index = [index]
        # print(type(index),index)
        sample = []
        idx = index
        # for idx in index:
        prompt = self.__generate_prompt__(idx)
        # prompt = self.first_line
        entry = self.dataset[idx]
        question = entry["question"]
        choices = entry["choices"]
        answer = entry["answer"]
        # answer = entry["answer"]+'.'+choices[ord(entry["answer"])-65]

        formatted_string = f"问题:{question}\n"
        formatted_string += "选项：\n"
        formatted_string += "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
        )
        formatted_string += f"\n答案: "
        prompt = prompt + "\n\n" + formatted_string
        human = prompt
        system = self.first_line
        gpt = answer
        messages = [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}]
        item = {"conversations": messages, "system": system}
        # self._tokenize(item, self.tokenizer)

        input_ids, labels = self._tokenize(
            copy.deepcopy(item),
            self.tokenizer)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
        )
        # self.cached_data_dict[i] = ret

        return ret
        # else:
        # prompt = self.__generate_prompt__(index)
        # entry = self.dataset[index]
        # question = entry['question']
        # choices = entry['choices']
        # answer = entry['answer']

        # formatted_string = f"问题:{question}\n"
        # formatted_string += '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        # formatted_string += f"\n答案: "
        # prompt = prompt + "\n\n" + formatted_string
        # sample = [prompt, answer]
        # return [sample]
    @staticmethod
    def _tokenize(item, tokenizer):
        roles = {"human": "user", "gpt": "assistant"}
        input_ids = []
        labels = []
        # if "instruction" in item and len(item["instruction"]) > 0:
        #     system = item["instruction"]
        # else:
        system = item["system"]
            # raise ValueError("instruction is empty")
        system = B_SYS + system + E_SYS
        # add system before the first content in conversations
        item["conversations"][0]['value'] = system + item["conversations"][0]['value']
        # item["input"] = system + item["input"]
        for i, turn in enumerate(item["conversations"]):
            role = turn['from']
            content = turn['value']
            content = content.strip()
            if role == 'human':
                content = f"{B_INST} {content} {E_INST} "
                content_ids = tokenizer.encode(content)
                labels += [IGNORE_TOKEN_ID] * (len(content_ids))
            else:
                # assert role == "gpt"
                content = f"{content} "
                content_ids = tokenizer.encode(content, add_special_tokens=False) + [tokenizer.eos_token_id]   # add_special_tokens=False remove bos token, and add eos at the end
                labels += content_ids
            input_ids += content_ids

        input_ids = input_ids[:tokenizer.model_max_length]
        labels = labels[:tokenizer.model_max_length]

        trunc_id = last_index(labels, IGNORE_TOKEN_ID) + 1
        input_ids = input_ids[:trunc_id]
        labels = labels[:trunc_id]
        if len(labels) == 0:
            return CEvalDataset._tokenize(CEvalDataset.dummy_message, tokenizer)
            assert False, "labels is empty"
        input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)
        labels = safe_ids(labels, tokenizer.vocab_size, IGNORE_TOKEN_ID)
        return input_ids, labels


class BUSTMDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        with open(ceval_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "BUSTM"
        self.first_line = "请根据提供的中文句子，判断它们是否属于同一语义：\n"
        self.item_size = item_size
        self.prompt_dict = {"1": "A", "0": "B"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = "请根据提供的中文句子，判断它们是否属于同一语义：\n"

        for i, sample in enumerate(samples):
            sentence1 = sample["sentence1"]
            sentence2 = sample["sentence2"]
            label = sample["label"]
            # Add the sample information to the prompt
            prompt += f"\n文本:"
            prompt += f"\n文本1: {sentence1}"
            prompt += f"\n文本2: {sentence2}"
            prompt += f"\nA:属于"
            prompt += f"\nB:不属于"
            prompt += f"\n答案:{self.prompt_dict[label]}\n"
        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = entry["label"]
        prompt += f"\n文本:"
        prompt += f"\n文本1: {sentence1}"
        prompt += f"\n文本2: {sentence2}"
        prompt += f"\nA:属于"
        prompt += f"\nB:不属于"
        prompt += f"\n答案:\n"
        sample = {"prompt": prompt, "answer": self.prompt_dict[answer]}
        return sample


class OCNLIDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        with open(ceval_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "OCNLI"
        self.first_line = "请根据下面的前提和假设句子进行推理，选择正确的关系：\n"
        self.item_size = item_size

        for data in self.dataset.copy():
            if data["label"] == "-":
                self.dataset.remove(data)

        self.prompt_dict = {"contradiction": "A", "neutral": "B", "entailment": "C"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            sentence1 = sample["sentence1"]
            sentence2 = sample["sentence2"]
            label = sample["label"]

            # Add the sample information to the prompt
            prompt += f"\n文本:"
            prompt += f"\n前提: {sentence1}"
            prompt += f"\n假设: {sentence2}"
            prompt += f"\nA:矛盾"
            prompt += f"\nB:中立"
            prompt += f"\nC:蕴含"
            prompt += f"\n答案:{self.prompt_dict[label]}\n"
        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = entry["label"]
        prompt += f"\n文本:"
        prompt += f"\n前提: {sentence1}"
        prompt += f"\n假设: {sentence2}"
        prompt += f"\nA:矛盾"
        prompt += f"\nB:中立"
        prompt += f"\nC:蕴含"
        prompt += f"\n答案:\n"
        sample = {"prompt": prompt, "answer": self.prompt_dict[answer]}
        return sample


class GAOKAO2023Dataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        self.dataset = json.load(open(ceval_path, "r", encoding="utf-8"))

        self.name = "GAOKAO2023"
        self.first_line = "这个任务是关于一组问题，请从给出的A、B、C、D四个选项中，选出其中的正确答案。请回答'A'或'B'或'C'或'D'\n"
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "问题：" + str(sample["question"]) + "\n"
            prompt += str(sample["choices"]) + "\n"
            prompt += "答案：" + str(sample["answer"][0]) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = entry["answer"][0]

        prompt += "问题：" + str(entry["question"]) + "\n"
        prompt += str(entry["choices"]) + "\n"
        prompt += "答案：" + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


class RAFTDataset(Dataset):
    """
    等待施工。。。
    """

    def __init__(self, ceval_path="", subset_name="", using_gpt=False, item_size=5):
        # subset_name=['ade_corpus_v2', 'banking_77', 'neurips_impact_statement_risks', 'one_stop_english', 'overruling', 'semiconductor_org_types', 'systematic_review_inclusion', 'tai_safety_research', 'terms_of_service', 'tweet_eval_hate', 'twitter_complaints']
        # self.dataset =datasets.load_dataset('ought/raft',subset_name)
        self.name = "RAFT"
        subset_name = [
            "ade_corpus_v2",
            "banking_77",
            "neurips_impact_statement_risks",
            "one_stop_english",
            "overruling",
            "semiconductor_org_types",
            "systematic_review_inclusion",
            "tai_safety_research",
            "terms_of_service",
            "tweet_eval_hate",
            "twitter_complaints",
        ]
        # self.first_line = "The following content is text labeling mission, label the final given text just like give samples.\n"
        self.first_line = "The following content is text labeling mission about "
        self.item_size = item_size

        self.sub2ind = {}
        self.sub2label = {}
        self.dataset = []
        i = 0
        for sub in subset_name:
            d = datasets.load_dataset("ought/raft", sub)
            lb = d["train"].features["Label"]
            self.sub2label[sub] = lb
            for split in d:
                for item in d[split]:
                    if item["Label"] == 0:
                        continue  # skip unlabeled
                    self.dataset.append(item)
                    self.sub2ind.setdefault(sub, []).append(i)
                    i += 1

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        sub_name = ""
        for n, inds in self.sub2ind.items():
            if ban_index in inds:
                sub_name = n
                break
        prompt = self.first_line+sub_name+".\n"
        labels = self.sub2label[sub_name].names # dataset['train'].features['Label'].names
        prompt_possible_answers = [f"{i}. {labels[i]}\n" for i in range(1, len(labels))]
        prompt += "".join(prompt_possible_answers) + "\n"
        # sub2ind[sub_name]
        inds = random.sample(self.sub2ind[sub_name], 5)
        for i in inds:
            item = self.dataset[i]
            item_prompt = ""
            for k, v in item.items():
                if k in ["ID", "id", "Id", "iD"]:
                    continue
                if k == "Label":
                    continue
                item_prompt += f"{k}: {v}\n"
            item_prompt += f"Label : {labels[item['Label']]}\n"
            prompt += item_prompt + "\n"
        return prompt, labels

    def __getitem__(self, index):
        prompt, labels = self.__generate_prompt__(index)
        item = self.dataset[index]
        item_prompt = ""
        for k, v in item.items():
            if k in ["ID", "id", "Id", "iD"]:
                continue
            if k == "Label":
                continue
            item_prompt += f"{k}: {v}\n"
        item_prompt += f"Label: \n"
        prompt += item_prompt
        answer = labels[item["Label"]]
        # sample = {"prompt": prompt, "answer": answer, "labels": labels}
        sample = {"prompt": prompt, "answer": answer}
        return sample
    def generate_labels(self,labels):
        choiceses=[]
        for label in labels:
            for subname in self.sub2label:
                if label in self.sub2label[subname].names:
                    # remove unlabeled
                    choiceses.append(self.sub2label[subname].names[1:])
                    break
        return choiceses
class TruthfulQADataset(Dataset):
    """TruthfulQA dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        self.dataset = datasets.load_dataset("truthful_qa", "multiple_choice")[
            "validation"
        ]

        self.name = "TruthfulQA"
        self.first_line = "The following are some questions about truthfulness. Please choose the most authentic answer from the options.\n\n"
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    # def __generate_prompt__(self, ban_index=-1):
    #     ind = random.sample(range(len(self.dataset)), self.item_size)
    #     samples = self.dataset.select(ind)
    #     # Initialize the prompt string
    #     prompt = self.first_line

    #     for i, sample in enumerate(samples):
    #         z = sample["mc1_targets"]
    #         prompt += str(sample["question"]) + "\n"
    #         prompt += "options: " + "\n"
    #         prompt += (
    #             "\n".join(
    #                 [
    #                     f"{a}. {c}"
    #                     for (a, c) in zip(ALPHABET[: len(z["choices"])], z["choices"])
    #                 ]
    #             )
    #             + "\n"
    #         )
    #         prompt += "Answer:" + ALPHABET[z["labels"].index(1)] + "\n"
    #         prompt += "\n"
    #     return prompt

    def __generate_prompt__(self, ban_index=-1):
        ind = random.sample(range(len(self.dataset)), self.item_size)
        samples = self.dataset.select(ind)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            z = sample["mc1_targets"]
            
            # Shuffle choices and adjust answer accordingly
            shuffled_indices = list(range(len(z["choices"])))
            random.shuffle(shuffled_indices)
            shuffled_choices = [z["choices"][i] for i in shuffled_indices]
            new_answer_index = shuffled_indices.index(z["labels"].index(1))

            prompt += str(sample["question"]) + "\n"
            prompt += "options: " + "\n"
            prompt += (
                "\n".join(
                    [f"{a}. {c}" for (a, c) in zip(ALPHABET[: len(shuffled_choices)], shuffled_choices)]
                )
                + "\n"
            )
            prompt += "Answer:" + ALPHABET[new_answer_index] + "\n"
            prompt += "\n"
        return prompt

    def __multi_choice_prompt__(self, ban_index=-1):
        ind = random.sample(range(len(self.dataset)), self.item_size)
        samples = self.dataset.select(ind)
        prompt = self.first_line

        for i, sample in enumerate(samples):
            z = sample["mc2_targets"]
            prompt += str(sample["question"]) + "\n"
            prompt += "Options: " + "\n"
            prompt += (
                "\n".join(
                    [
                        f"{a}. {c}"
                        for (a, c) in zip(ALPHABET[: len(z["choices"])], z["choices"])
                    ]
                )
                + "\n"
            )
            ans = []
            for j, v in enumerate(sample["mc2_targets"]["labels"]):
                if v == 1:
                    ans.append(ALPHABET[int(j)])
            prompt += "Answer:" + ", ".join(ans) + "\n"
            prompt += "\n"
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.first_line
        # prompt = self.__multi_choice_prompt__(idx)
        entry = self.dataset[idx]
        z = entry["mc1_targets"]
        # z=entry['mc2_targets']
        answer = ALPHABET[z["labels"].index(1)]
        z = z["choices"]
        prompt += str(entry["question"]) + "\n"
        prompt += "Options: " + "\n"
        prompt += (
            "\n".join([f"{a}. {c}" for (a, c) in zip(ALPHABET[: len(z)], z)]) + "\n"
        )
        prompt += "Answer: " + "\n"
        prompt += "\n"
        # 数据集里都是A，需要做一下shuffle
        sample = {"prompt": prompt, "answer": answer}
        return sample


class EPRSTMTDataset(Dataset):
    """EPRSTMT dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        with open(ceval_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "EPRSTMT"
        self.first_line = "判断下列评价是好评还是差评：\n"
        self.item_size = item_size
        self.prompt_dict = {"Positive": "A", "Negative": "B"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            sentence = sample["sentence"]
            label = sample["label"]
            # 组合samples
            prompt += f"\n文本:"
            prompt += f"\n评价: {sentence}"
            prompt += f"\nA:好评"
            prompt += f"\nB:差评"
            prompt += f"\n答案:{self.prompt_dict[label]}\n"
            prompt += "\n"
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        question = self.dataset[idx]
        # z=entry['mc2_targets']
        sentence = question["sentence"]
        prompt += f"\n文本:"
        prompt += f"\n评价: {sentence}"
        prompt += f"\nA:好评"
        prompt += f"\nB:差评"
        prompt += f"\n答案:\n"

        sample = {"prompt": prompt, "answer": self.prompt_dict[question["label"]]}
        return sample


class TNEWSDataset(Dataset):
    """TNEWS dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        with open(ceval_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "TNEWS"
        self.first_line = "请判断出下列新闻的分类：\n"
        self.item_size = item_size
        self.prompt_dict = {
            "news_house": "A",
            "news_entertainment": "B",
            "news_sports": "C",
            "news_game": "D",
            "news_military": "E",
            "news_culture": "F",
            "news_finance": "G",
            "news_agriculture": "H",
            "news_world": "I",
            "news_travel": "J",
            "news_tech": "K",
            "news_story": "L",
            "news_stock": "M",
            "news_edu": "N",
            "news_car": "O",
        }

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line
        for i, sample in enumerate(samples):
            sentence = sample["sentence"]
            label = sample["label_desc"]
            keywords = sample["keywords"]
            # print(keywords)
            # 组合samples
            prompt += f"\n文本:"
            prompt += f"\n新闻标题: {sentence}"
            prompt += f"\n关键词: {keywords}"
            prompt += f"\nA:家庭新闻"
            prompt += f"\nB:娱乐新闻"
            prompt += f"\nC:体育新闻"
            prompt += f"\nD:游戏新闻"
            prompt += f"\nE:军事新闻"
            prompt += f"\nF:文化新闻"
            prompt += f"\nG:经济新闻"
            prompt += f"\nH:农业新闻"
            prompt += f"\nI:环球新闻"
            prompt += f"\nJ:旅行新闻"
            prompt += f"\nK:技术新闻"
            prompt += f"\nL:故事新闻"
            prompt += f"\nM:金融新闻"
            prompt += f"\nN:教育新闻"
            prompt += f"\nO:汽车新闻"
            prompt += f"\n答案:{self.prompt_dict[label]}\n"
            prompt += "\n"
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        question = self.dataset[idx]
        # z=entry['mc2_targets']
        sentence = question["sentence"]
        label = question["label_desc"]
        keywords = question["keywords"]
        prompt += f"\n文本:"
        prompt += f"\n新闻标题: {sentence}"
        prompt += f"\n关键词: {keywords}"
        prompt += f"\nA:家庭新闻"
        prompt += f"\nB:娱乐新闻"
        prompt += f"\nC:体育新闻"
        prompt += f"\nD:游戏新闻"
        prompt += f"\nE:军事新闻"
        prompt += f"\nF:文化新闻"
        prompt += f"\nG:经济新闻"
        prompt += f"\nH:农业新闻"
        prompt += f"\nI:环球新闻"
        prompt += f"\nJ:旅行新闻"
        prompt += f"\nK:技术新闻"
        prompt += f"\nL:故事新闻"
        prompt += f"\nM:金融新闻"
        prompt += f"\nN:教育新闻"
        prompt += f"\nO:汽车新闻"
        prompt += f"\n答案:\n"
        sample = {"prompt": prompt, "answer": self.prompt_dict[label]}
        return sample


class IMDBDataset(Dataset):
    """IMDB dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        self.dataset = datasets.load_dataset("imdb")["test"]

        self.name = "IMDB"
        self.first_line = "In this task, you will be presented with some text. Please determine whether the text is positive or negative.Please answer with 'Positive' or 'Negative'.\n"
        self.item_size = item_size
        self.prompt_dict = {1: "Positive", 0: "Negative"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        ind = random.sample(range(len(self.dataset)), self.item_size)
        samples = self.dataset.select(ind)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            prompt += str(sample["text"]) + "\n"
            prompt += "Answer: " + self.prompt_dict[sample["label"]] + "\n"
            prompt += "\n"
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        sample = self.dataset[idx]
        # z=entry['mc2_targets']
        answer = self.prompt_dict[sample["label"]]
        prompt += str(sample["text"]) + "\n"
        prompt += "Answer: " + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample
    def generate_labels(self,labels):
        choiceses=[]
        for label in labels:
            choiceses.append(list(self.prompt_dict.values()))
        return choiceses

class BoolQDataset(Dataset):
    """BoolQ dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        dataset = datasets.load_dataset("boolq")

        self.name = "BoolQ"
        self.prompt_heads = [
            "Read the passage and answer the following true/false question.",
            "Determine the correctness of the statement based on the given passage.",
            "Decide whether the statement is true or false according to the passage.",
            "After reading the passage, indicate if the statement is true or false.",
            "Choose whether the statement is true or false based on the provided passage.",
            "Based on the passage, determine if the statement is true or false.",
            "Confirm the truthfulness of the statement based on the passage.",
        ]

        # 数据集文件是arrow文件，所以需要用datasets.load_from_disk，folder_path是数据集的文件夹路径
        self.item_size = item_size
        # self.prompt_dict = {1:'Positive', 0:'Negative'}
        self.choice = ["True", "False"]
        train_content = []
        # 将数据集里的所有题目填进一个列表中
        for ele in dataset:
            for k in range(len(dataset[ele])):
                train_content.append(dataset[ele][k])
        self.dataset = train_content

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        train_sample = random.sample(self.dataset, self.item_size)
        prompt = [random.choice(self.prompt_heads) + "\n"]
        prompt_choice = "A. " + self.choice[0] + "\nB. " + self.choice[1] + "\n"
        for i, item in enumerate(train_sample):
            FLAG = str(item.get("answer", ""))
            if FLAG == "True":
                FLAG = "A"
            elif FLAG == "False":
                FLAG = "B"
            prompt_item = (
                "\nPassage: "
                + item["passage"]
                + "\nQuestion: "
                + item["question"]
                + "?\n"
                + prompt_choice
                + "\nAnswer: "
                + FLAG
                + "\n"
            )
            prompt.append(prompt_item)
        prompt = "".join(prompt)
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        sample = self.dataset[idx]
        # z=entry['mc2_targets']
        # answer = self.prompt_dict[sample['answer']]
        FLAG = str(sample.get("answer", ""))
        if FLAG == "True":
            FLAG = "A"
        elif FLAG == "False":
            FLAG = "B"
        answer = FLAG
        prompt_choice = "A. " + self.choice[0] + "\nB. " + self.choice[1] + "\n"
        prompt += (
            "\nPassage: "
            + sample["passage"]
            + "\nQuestion: "
            + sample["question"]
            + "?\n"
            + prompt_choice
            + "\nAnswer: "
            + "\n"
        )

        sample = {"prompt": prompt, "answer": answer}
        return sample


class MMLUDataset(Dataset):
    """MMLU dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        # dataset = load_dataset("tasksource/mmlu")
        dataset_name = "tasksource/mmlu"
        courses = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]
        self.name = "MMLU"
        self.prompt_heads = [
            "The following are multiple-choice questions with their respective answer choices.",
            "Test your knowledge with the following multiple-choice questions and select the correct answer.",
            "Answer the following multiple-choice questions by selecting the appropriate options.",
            "This prompt presents a set of multiple-choice questions for you to answer.",
            "Choose the correct option for the following multiple-choice questions.",
            "Select the most suitable answer from the options provided for multiple-choice question.",
            "Test your knowledge with the following multiple-choice questions by selecting the correct answers.",
            "This prompt presents a set of multiple-choice questions covering different fields of knowledge.",
            "Answer each of the following multiple-choice questions by selecting the most appropriate option.",
            "Evaluate your knowledge by selecting the correct answer for each of the following multiple-choice questions.",
            "This prompt contains multiple-choice questions from a wide range of subjects.",
            "Select the appropriate option for each of the following multiple-choice questions.",
            "Choose the correct option from the given choices for each multiple-choice question.",
        ]
        # self.prompt_heads=[
        #     "The following are multiple choice questions about "
        # ]

        # 数据集文件是arrow文件，所以需要用datasets.load_from_disk，folder_path是数据集的文件夹路径
        self.item_size = item_size
        # self.prompt_dict = {1:'Positive', 0:'Negative'}
        self.choice = ["True", "False"]
        train_content = []
        # 将数据集里的所有题目填进一个列表中
        for sub in courses:
            dataset = load_dataset(dataset_name, sub)
            for ele in dataset:
                for k in range(len(dataset[ele])):
                    train_content.append(dataset[ele][k])
        self.dataset = train_content

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        train_sample = random.sample(self.dataset, self.item_size)
        prompt = [random.choice(self.prompt_heads) + "\n\n"]
        for item in train_sample:
            choice = item["choices"]  # list of choices, number of choices varies
            # choice in prompt should have prefix of ABCE according to the number of choices
            prompt_choice = []
            for i in range(len(choice)):
                prompt_choice.append(f"{chr(65+i)}. {choice[i]}")
            prompt_choice = "\n".join(prompt_choice)
            Flag = ""

            Choice = []
            for i in range(len(choice)):
                Choice.append(f"{chr(65+i)}")
            if item.get("answer", "") != "":
                Flag = Choice[item.get("answer", "")]
            prompt_item = (
                f"Question: {item['question']}?\n{prompt_choice}\nAnswer: {Flag}\n\n"
            )
            prompt.append(prompt_item)

        prompt = "".join(prompt)
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        sample = self.dataset[idx]
        choice = sample["choices"]  # list of choices, number of choices varies
        # choice in prompt should have prefix of ABCE according to the number of choices
        prompt_choice = []
        for i in range(len(choice)):
            prompt_choice.append(f"{chr(65+i)}. {choice[i]}")
        prompt_choice = "\n".join(prompt_choice)
        Flag = ""

        Choice = []
        for i in range(len(choice)):
            Choice.append(f"{chr(65+i)}")
        if sample.get("answer", "") != "":
            Flag = Choice[sample.get("answer", "")]
        prompt_item = f"Question: {sample['question']}?\n{prompt_choice}\nAnswer: \n\n"
        prompt += prompt_item

        sample = {"prompt": prompt, "answer": Flag}
        return sample


class CMMLUDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        with open(ceval_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        self.name = "CMMLU"
        self.first_line = "下面是一组问题，每个问题均有四个选项，请选出正确答案。\n"
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "问题：" + str(sample["Question"]) + "\n"
            prompt += "A." + str(sample["choices"][0]) + "\n"
            prompt += "B." + str(sample["choices"][1]) + "\n"
            prompt += "C." + str(sample["choices"][2]) + "\n"
            prompt += "D." + str(sample["choices"][3]) + "\n"
            prompt += "答案：" + str(sample["Answer"]) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = entry["Answer"]

        prompt += "问题：" + str(entry["Question"]) + "\n"
        prompt += "A." + str(entry["choices"][0]) + "\n"
        prompt += "B." + str(entry["choices"][1]) + "\n"
        prompt += "C." + str(entry["choices"][2]) + "\n"
        prompt += "D." + str(entry["choices"][3]) + "\n"
        prompt += "答案：" + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


class ChIDDataset(Dataset):
    """ChID dataset
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        self.name = "ChID"
        with open(ceval_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.first_line = (
            "在这个任务中，你将面对一些不完整的句子，其中句子中成语被'#idiom#'取代，以成语完形填空形式实现，从给定的七个选项,选择正确答案：\n"
        )
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "不完整的句子：" + str(sample["content"]) + "\n"
            prompt += "选项：" + "\n"
            prompt += "A." + str(sample["candidates"][0]) + "\n"
            prompt += "B." + str(sample["candidates"][1]) + "\n"
            prompt += "C." + str(sample["candidates"][2]) + "\n"
            prompt += "D." + str(sample["candidates"][3]) + "\n"
            prompt += "E." + str(sample["candidates"][4]) + "\n"
            prompt += "F." + str(sample["candidates"][5]) + "\n"
            prompt += "G." + str(sample["candidates"][6]) + "\n"

            prompt += "答案：" + chr(ord("A") + int(sample["answer"])) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = chr(ord("A") + int(entry["answer"]))

        prompt += "不完整的句子：" + str(entry["content"]) + "\n"
        prompt += "选项:" + "\n"
        prompt += "A." + str(entry["candidates"][0]) + "\n"
        prompt += "B." + str(entry["candidates"][1]) + "\n"
        prompt += "C." + str(entry["candidates"][2]) + "\n"
        prompt += "D." + str(entry["candidates"][3]) + "\n"
        prompt += "E." + str(entry["candidates"][4]) + "\n"
        prompt += "F." + str(entry["candidates"][5]) + "\n"
        prompt += "G." + str(entry["candidates"][6]) + "\n"
        prompt += "答案：" + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


class CSLDataset(Dataset):
    """CSL dataset
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        self.name = "CSL"
        with open(ceval_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.first_line = "在此任务中，将给出一段摘要与几个关键词，根据给出的摘要与关键词的关系，判断关键词是真实还是伪造，关键词为真实时请回答'真实'，关键词为伪造时请回答'伪造'：\n"
        self.item_size = item_size
        self.prompt_dict = {"1": "真实", "0": "伪造"}
        self.choice_dict = {"1": "A", "0": "B"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line
        choices=["A","B"]
        indexes=["0","1"]
        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "摘要：" + str(sample["abst"]) + "\n"
            prompt += "关键词：" + str(sample["keyword"]) + "\n"
            random.shuffle(indexes)
            prompt += choices[indexes.index("0")] + "." + str(self.prompt_dict["0"]) + "\n"
            prompt += choices[indexes.index("1")] + "." + str(self.prompt_dict["1"]) + "\n"
            # prompt += "答案：" + str(self.prompt_dict[str(sample["label"])]) + "\n"
            prompt += "答案：" + choices[indexes.index(str(sample["label"]))] + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        choices=["A","B"]
        indexes=["0","1"]
        random.shuffle(indexes)
 
        # answer = str(self.prompt_dict[str(entry["label"])])
        answer = choices[indexes.index(str(entry["label"]))]

        prompt += "摘要：" + str(entry["abst"]) + "\n"
        prompt += "关键词：" + str(entry["keyword"]) + "\n"
        prompt += choices[indexes.index("0")] + "." + str(self.prompt_dict["0"]) + "\n"
        prompt += choices[indexes.index("1")] + "." + str(self.prompt_dict["1"]) + "\n"
        prompt += "答案：" + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


class CLUEWSCDataset(Dataset):
    """CLUEWSC dataset
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        self.name = "CLUEWSC"
        with open(ceval_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.first_line = "根据下面这段句子，判断指定位置的代词是否指代指定位置的名词,如果是请回答'是',如果不是请回答'否'\n"
        self.item_size = item_size
        self.prompt_dict = {"true": "是", "false": "否"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "句子：" + str(sample["text"]) + "\n"
            prompt += (
                "位于指定位置("
                + str(sample["target"]["span2_index"])
                + ")的代词："
                + str(sample["target"]["span2_text"])
                + "\n"
            )
            prompt += (
                "位于指定位置("
                + str(sample["target"]["span1_index"])
                + ")的名词："
                + str(sample["target"]["span1_text"])
                + "\n"
            )
            prompt += "答案：" + str(self.prompt_dict[str(sample["label"])]) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = str(self.prompt_dict[str(entry["label"])])

        prompt += "" + str(entry["text"]) + "\n"
        prompt += (
            "位于指定位置("
            + str(entry["target"]["span2_index"])
            + ")的代词："
            + str(entry["target"]["span2_text"])
            + "\n"
        )
        prompt += (
            "位于指定位置("
            + str(entry["target"]["span1_index"])
            + ")的名词："
            + str(entry["target"]["span1_text"])
            + "\n"
        )

        prompt += "答案：" + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample
