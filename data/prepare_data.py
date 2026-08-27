from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset

DATASET_NAME = "newfacade/LeetCodeDataset"
LCB_DATASET = "livecodebench/code_generation_lite"
LCB_RELEASE = "release_v6"
CF_DATASET = "open-r1/codeforces"
CF_CONFIG = "verifiable-prompts"
CF_LANGUAGE = "python"
DEFAULT_NGRAM_SIZE = 12


@dataclass(frozen=True)
class LeetCodeSample:
    task_id: str
    question_id: int
    difficulty: str
    problem_description: str
    starter_code: str
    query: str
    response: str
    prompt: str
    completion: str
    entry_point: str
    test: str


class LeetCodeRLVRDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset) -> None:
        self._dataset = hf_dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> LeetCodeSample:
        row = self._dataset[idx]
        return LeetCodeSample(
            task_id=row["task_id"],
            question_id=row["question_id"],
            difficulty=row["difficulty"],
            problem_description=row["problem_description"],
            starter_code=row["starter_code"],
            query=row["query"],
            response=row["response"],
            prompt=row["prompt"],
            completion=row["completion"],
            entry_point=row["entry_point"],
            test=row["test"],
        )


def normalize_string(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def word_ngrams(text: str, ngram_size: int) -> list[str]:
    words = normalize_string(text).split()
    if not words:
        return []
    if len(words) < ngram_size:
        return [" ".join(words)]
    return [" ".join(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]


def build_ngram_lookup(documents: list[str], ngram_size: int) -> set[str]:
    lookup: set[str] = set()
    for document in documents:
        lookup.update(word_ngrams(document, ngram_size))
    return lookup


def has_ngram_overlap(text: str, ngram_lookup: set[str], ngram_size: int) -> bool:
    return any(ngram in ngram_lookup for ngram in word_ngrams(text, ngram_size))


@lru_cache(maxsize=4)
def _load_lcb_eval() -> tuple[str, ...]:
    dataset = load_dataset(LCB_DATASET, LCB_RELEASE, split="test", trust_remote_code=True)
    return tuple(dataset["question_content"])


@lru_cache(maxsize=4)
def _load_cf_eval_descriptions() -> tuple[str, ...]:
    dataset = load_dataset(CF_DATASET, CF_CONFIG, split="test", trust_remote_code=True)
    descriptions = tuple(
        row["description"]
        for row in dataset
        if row["language"] == CF_LANGUAGE
    )
    return descriptions


@lru_cache(maxsize=8)
def _eval_ngram_lookups(ngram_size: int) -> tuple[set[str], set[str]]:
    lcb_lookup = build_ngram_lookup(list(_load_lcb_eval()), ngram_size)
    cf_lookup = build_ngram_lookup(list(_load_cf_eval_descriptions()), ngram_size)
    return lcb_lookup, cf_lookup


def is_contaminated(problem_description: str, *, ngram_size: int = DEFAULT_NGRAM_SIZE, lcb_lookup: set[str] | None = None, cf_lookup: set[str] | None = None) -> bool:
    if lcb_lookup is None or cf_lookup is None:
        lcb_lookup, cf_lookup = _eval_ngram_lookups(ngram_size)

    return has_ngram_overlap(problem_description, lcb_lookup, ngram_size) or has_ngram_overlap(problem_description, cf_lookup, ngram_size)


def filter_contaminated(dataset: Dataset, *, ngram_size: int = DEFAULT_NGRAM_SIZE) -> Dataset:
    lcb_lookup, cf_lookup = _eval_ngram_lookups(ngram_size)

    def keep_row(row: dict[str, Any]) -> bool:
        return not is_contaminated(row["problem_description"], ngram_size=ngram_size, lcb_lookup=lcb_lookup, cf_lookup=cf_lookup)

    return dataset.filter(keep_row)


def load_leetcode_dataset(split: str = "train", *, decontaminate: bool = True, ngram_size: int = DEFAULT_NGRAM_SIZE) -> Dataset:
    dataset = load_dataset(DATASET_NAME, split=split)
    if decontaminate:
        dataset = filter_contaminated(dataset, ngram_size=ngram_size)
    return dataset


def collate_leetcode_samples(batch: list[LeetCodeSample]) -> dict[str, list[Any]]:
    if not batch:
        return {}

    keys = batch[0].__dataclass_fields__.keys()
    return {key: [getattr(sample, key) for sample in batch] for key in keys}


def get_dataloader(
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    *,
    decontaminate: bool = True,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    **loader_kwargs: Any,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"

    dataset = LeetCodeRLVRDataset(load_leetcode_dataset(split, decontaminate=decontaminate, ngram_size=ngram_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_leetcode_samples, **loader_kwargs)


def sample_to_dict(sample: LeetCodeSample) -> dict[str, Any]:
    return asdict(sample)
