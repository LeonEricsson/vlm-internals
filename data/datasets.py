"""
Source: https://github.com/shiqichen17/AdaptVis/blob/main/dataset_zoo/aro_datasets.py
"""

import pdb
import os
import json
import subprocess
import re
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url


class VSR(Dataset):
    def __init__(
        self,
        image_preprocess=None,
        root_dir="data",
        max_words=30,
        split="test",
        image_perturb_fn=None,
        download=False,
    ):
        """
        COCO Order Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: not used; for compatibility.
        download: Whether to download the dataset if it does not exist.
        """

        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print("Directory for COCO could not be found!")
            if download:
                print("Downloading COCO now.")
                # pdb.set_trace()
                self.download()
                # pdb.set_trace()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        urls = {
            "val": "https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/raw/main/val.jsonl",
            "test": "https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/raw/main/test.jsonl",
        }
        filenames = {"val": "val.jsonl", "test": "test.jsonl"}
        download_url(urls[split], root_dir)

        import json

        def load_jsonl(file_path):
            data = []
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line))
            return data

        self.image_root = os.path.join(root_dir, "train2017")
        if not os.path.exists(self.image_root):
            print("Image Directory for VG_Relation could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        self.annotation = load_jsonl(os.path.join(root_dir, filenames[split]))
        self.image_preprocess = image_preprocess

        self.test_cases = []

        for img_id, ann in tqdm(enumerate(self.annotation)):
            test_case = {}

            image_link = re.search(r"\.org/(.*)", ann["image_link"]).group(1)

            test_case["image"] = image_link
            test_case["caption"] = ann["caption"]
            test_case["label"] = ann["label"]
            # pdb.set_trace()
            self.test_cases.append(test_case)

    def __len__(self):
        return len(self.test_cases)

    def get_labels(self):
        array_shape = (len(self.test_cases), 1, 1)
        labels = np.zeros(array_shape)
        for index in range(len(self.test_cases)):
            test_case = self.test_cases[index]
            # image_path = os.path.join(self.image_root, test_case["image"])
            labels[index] = test_case["label"]

        # pdb.set_trace()
        return labels

    def __getitem__(self, index):
        test_case = self.test_cases[index]
        # image_path = os.path.join(self.image_root, test_case["image"])
        image_path = os.path.join("data", test_case["image"])

        image = Image.open(image_path).convert("RGB").resize((2300, 4096))
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        item = edict(
            {
                "image_options": [image],
                "caption_options": [test_case["caption"]],
                "labels": [test_case["label"]],
            }
        )

        return item

    def download(self):
        import subprocess

        os.makedirs(self.root_dir, exist_ok=True)

        subprocess.call(
            ["wget", "http://images.cocodataset.org/zips/train2017.zip"],
            cwd=self.root_dir,
        )
        subprocess.call(["unzip", "train2017.zip"], cwd=self.root_dir)

    def evaluate_scores(self, model_name, scores, labels, path, dataset):
        # if model_name=='llava1.6_add_attn' or model_name=='llava1.5_add_attn':
        if dataset == "VSR":
            path_ = os.path.join(path, "res.json")
            import json

            data = {"dataset": dataset, "model": model_name, "scores": scores}
            with open(path_, "a+") as file:
                json.dump(data, file)

        else:
            from sklearn.metrics import roc_auc_score

            score_flat = scores.flatten()
            label_flat = labels.flatten()
            print(
                f"acc: {sum([1 for x, y in zip(score_flat, label_flat) if x == y]) / len(label_flat)}"
            )
            # pdb.set_trace()
            TP = np.sum(score_flat[np.where(label_flat == 1)])
            P = np.sum(label_flat)
            TN = len(score_flat[np.where(label_flat == 0)]) - np.sum(
                score_flat[np.where(label_flat == 0)]
            )
            N = len(label_flat) - np.sum(label_flat)
            print(f"TP,P,TN,N:{TP, P, TN, N}")
            recall = np.sum(score_flat[np.where(label_flat == 1)]) / np.sum(label_flat)
            precision = (
                len(score_flat[np.where(label_flat == 0)])
                - np.sum(score_flat[np.where(label_flat == 0)])
            ) / (len(label_flat) - np.sum(label_flat))
            print(f"recall: {recall}")
            print(f"precision: {precision}")
            print(f"f1: {2 * recall * precision / (recall + precision)}")

            auc = roc_auc_score(label_flat, score_flat)
            print(f"auc:{auc}")
            import json

            path_ = os.path.join(path, "res.json")
            data = {"dataset": dataset, "mode": mode, "AUROC": auc * 100}
            with open(path_, "a+") as file:
                json.dump(data, file)
            return auc


def get_vsr(
    image_preprocess,
    image_perturb_fn,
    text_perturb_fn,
    max_words=30,
    download=False,
    root_dir="data",
    split="test",
):
    return VSR(
        root_dir=root_dir,
        split=split,
        image_preprocess=image_preprocess,
        image_perturb_fn=image_perturb_fn,
        max_words=max_words,
        download=download,
    )
