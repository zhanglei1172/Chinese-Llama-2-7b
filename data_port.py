import os
import sys
from tqdm import tqdm

from flageval_datasets import (
    CEvalDataset,
    BUSTMDataset,
    OCNLIDataset,
    GAOKAO2023Dataset,
    TruthfulQADataset,
    EPRSTMTDataset,
    TNEWSDataset,
    CMMLUDataset,
    ChIDDataset,
    CSLDataset,
    CLUEWSCDataset,
    RAFTDataset,
    TruthfulQADataset,
    IMDBDataset,
    BoolQDataset,
    MMLUDataset,
    huggingface_datasets,
)
import torch
from torch.utils.data import ConcatDataset

ALL_DATASET = [
    "BoolQ",
    "MMLU",
    "TruthfulQA",
    "IMDB",
    "RAFT",
    "Chinese_MMLU",
    "C-Eval",
    "GAOKAO2023",
    "CSL",
    "ChID",
    "CLUEWSC",
    "EPRSTMT",
    "TNEWS",
    "OCNLI",
    "BUSTM",
]
ALL_PATH_DIC = {
    "C-Eval": "/mnt/nfs-share/FlagEvalChat/data/process/ceval_json/test",
    "BUSTM": "/data/LLM/flagevalmock/BUSTM",
    "OCNLI": "/data/LLM/flagevalmock/OCNLI",
    "GAOKAO2023": "/data/LLM/flagevalmock/GAOKAO2023",
    "EPRSTMT": "/data/LLM/flagevalmock/EPRSTMT",
    "TNEWS": "/data/LLM/flagevalmock/TNEWS",
    "Chinese_MMLU": "/data/LLM/flagevalmock/cmmlu/dev",  # test 无标签
    "ChID": "/data/LLM/flagevalmock/chid",
    "CSL": "/data/LLM/flagevalmock/csl",
    "CLUEWSC": "/data/LLM/flagevalmock/cluewsc",
}
CLASSES = 24
ALL_CLASSES = {
    "BoolQ": 2,
    "MMLU": 4,
    "TruthfulQA": 4,
    "IMDB": 2,
    "RAFT": 26, # 多个子集，选项数目不定
    "Chinese_MMLU":4,
    "C-Eval":4,
    "GAOKAO2023":4,
    "CSL":2,
    "ChID":7,
    "CLUEWSC":2, # 形式很奇怪，看看咋回事
    "EPRSTMT":2,
    "TNEWS":15,
    "OCNLI":3, # Neural, Entailment, Contradiction
    "BUSTM":2,

}
ALL_NEW_TOKENS = {
    "BoolQ": 1,
    "MMLU": 1,
    "TruthfulQA": 1,
    "IMDB": 4,
    "RAFT": 8, # 多个子集，选项数目不定
    "Chinese_MMLU":1,
    "C-Eval":1,
    "GAOKAO2023":1,
    "CSL":1,
    "ChID":1,
    "CLUEWSC":1, # 形式很奇怪，看看咋回事
    "EPRSTMT":4,
    "TNEWS":4,
    "OCNLI":4, # Neural, Entailment, Contradiction
    "BUSTM":4,

}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_dataset(dataset_name: str = ""):
    assert dataset_name in ALL_DATASET
    if dataset_name in huggingface_datasets:
        if dataset_name == "RAFT":
            dataset = RAFTDataset()
        elif dataset_name == "TruthfulQA":
            dataset = TruthfulQADataset()
        elif dataset_name == "IMDB":
            dataset = IMDBDataset()
        elif dataset_name == "BoolQ":
            dataset = BoolQDataset()
        elif dataset_name == "MMLU":
            dataset = MMLUDataset()
    else:
        datasets = []
        for root, dirs, files in os.walk(ALL_PATH_DIC[dataset_name]):
            for file in tqdm(files):
                valjson = os.path.join(root, file)
                try:
                    if dataset_name == "C-Eval":
                        dataset = CEvalDataset(valjson)
                    elif dataset_name == "BUSTM":
                        dataset = BUSTMDataset(valjson)
                    elif dataset_name == "OCNLI":
                        dataset = OCNLIDataset(valjson)
                    elif dataset_name == "GAOKAO2023":
                        dataset = GAOKAO2023Dataset(valjson)
                    elif dataset_name == "EPRSTMT":
                        dataset = EPRSTMTDataset(valjson)
                    elif dataset_name == "TNEWS":
                        dataset = TNEWSDataset(valjson)
                    elif dataset_name == "Chinese_MMLU":
                        dataset = CMMLUDataset(valjson)
                    elif dataset_name == "ChID":
                        dataset = ChIDDataset(valjson)
                    elif dataset_name == "CSL":
                        dataset = CSLDataset(valjson)
                    elif dataset_name == "CLUEWSC":
                        dataset = CLUEWSCDataset(valjson)
                    if len(dataset) == 0:
                        continue
                except Exception as e:
                    print(valjson)
                    print(e)
                    exit()
                datasets.append(dataset)
        dataset = ConcatDataset(datasets=datasets)
    return dataset

get_dataset("C-Eval")
