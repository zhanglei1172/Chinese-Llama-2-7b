import json
import random
import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

huggingface_datasets = ["RAFT", "TruthfulQA", "IMDB", "BoolQ", "MMLU"]


class CEvalDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
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
        sample = {"prompt": prompt, "answer": answer}
        # sample.append([prompt, answer])
        return sample
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
