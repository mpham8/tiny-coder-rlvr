import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import prepare_data
from data.prepare_data import (
    DATASET_NAME,
    DEFAULT_NGRAM_SIZE,
    LeetCodeRLVRDataset,
    build_ngram_lookup,
    collate_leetcode_samples,
    filter_contaminated,
    get_dataloader,
    has_ngram_overlap,
    is_contaminated,
    load_leetcode_dataset,
    word_ngrams,
)


class NgramUtilTest(unittest.TestCase):
    def test_word_ngrams_pads_short_text(self):
        self.assertEqual(word_ngrams("one two three", 8), ["one two three"])

    def test_has_ngram_overlap(self):
        lookup = build_ngram_lookup(["the quick brown fox jumps over the lazy dog"], ngram_size=3)
        self.assertTrue(
            has_ngram_overlap("a quick brown fox runs through the forest", lookup, ngram_size=3)
        )
        self.assertFalse(
            has_ngram_overlap("completely unrelated content here", lookup, ngram_size=3)
        )

    def test_higher_ngram_is_stricter(self):
        shared_prefix = "Given an array of integers nums and an integer target"
        train_text = shared_prefix + " return indices using a hash map."
        eval_text = shared_prefix + " return indices with a two pointer approach on sorted input."
        loose = build_ngram_lookup([eval_text], ngram_size=8)
        strict = build_ngram_lookup([eval_text], ngram_size=20)
        self.assertTrue(has_ngram_overlap(train_text, loose, ngram_size=8))
        self.assertFalse(has_ngram_overlap(train_text, strict, ngram_size=20))


class PrepareDataTest(unittest.TestCase):
    def setUp(self):
        prepare_data._load_lcb_eval.cache_clear()
        prepare_data._load_cf_eval_descriptions.cache_clear()
        prepare_data._eval_ngram_lookups.cache_clear()

    def test_dataset_name(self):
        self.assertEqual(DATASET_NAME, "newfacade/LeetCodeDataset")

    def test_default_ngram_size(self):
        self.assertEqual(DEFAULT_NGRAM_SIZE, 12)

    def test_load_train_split_without_decontamination(self):
        dataset = load_leetcode_dataset(split="train", decontaminate=False)
        self.assertGreater(len(dataset), 0)
        self.assertIn("task_id", dataset.features)
        self.assertIn("query", dataset.features)
        self.assertIn("test", dataset.features)

    def test_torch_dataset_item(self):
        hf_dataset = load_leetcode_dataset(split="train", decontaminate=False)
        dataset = LeetCodeRLVRDataset(hf_dataset)
        sample = dataset[0]

        self.assertEqual(sample.task_id, hf_dataset[0]["task_id"])
        self.assertTrue(sample.problem_description)
        self.assertTrue(sample.query)
        self.assertTrue(sample.prompt)
        self.assertTrue(sample.completion)
        self.assertTrue(sample.entry_point)
        self.assertTrue(sample.test)

    def test_collate_batch(self):
        hf_dataset = load_leetcode_dataset(split="train", decontaminate=False)
        dataset = LeetCodeRLVRDataset(hf_dataset)
        batch = collate_leetcode_samples([dataset[0], dataset[1]])

        self.assertEqual(len(batch["task_id"]), 2)
        self.assertEqual(batch["task_id"][0], hf_dataset[0]["task_id"])
        self.assertEqual(batch["task_id"][1], hf_dataset[1]["task_id"])

    def test_filter_contaminated_with_mocked_eval(self):
        from datasets import Dataset as HFDataset

        train = HFDataset.from_dict(
            {
                "task_id": ["keep-me", "drop-me"],
                "question_id": [1, 2],
                "problem_description": [
                    "Find the longest unique substring in a custom alphabet.",
                    "Given an array of integers nums and an integer target, return indices.",
                ],
            }
        )
        lcb_lookup = build_ngram_lookup(
            ["Given an array of integers nums and an integer target, return indices of the two numbers."],
            ngram_size=8,
        )

        with patch.object(
            prepare_data,
            "_eval_ngram_lookups",
            return_value=(lcb_lookup, set()),
        ):
            filtered = filter_contaminated(train, ngram_size=8)

        self.assertEqual(filtered["task_id"], ["keep-me"])

    def test_is_contaminated_respects_custom_lookups(self):
        lookup = build_ngram_lookup(["alpha beta gamma delta epsilon zeta eta theta"], ngram_size=3)
        self.assertTrue(
            is_contaminated(
                "prefix alpha beta gamma delta suffix",
                ngram_size=3,
                lcb_lookup=lookup,
                cf_lookup=set(),
            )
        )

    def test_dataloader(self):
        loader = get_dataloader(
            split="train",
            batch_size=2,
            shuffle=False,
            decontaminate=False,
        )
        batch = next(iter(loader))

        self.assertEqual(len(batch["task_id"]), 2)
        self.assertIn("problem_description", batch)
        self.assertIn("entry_point", batch)

    def test_decontaminated_train_is_smaller_than_raw(self):
        raw = load_leetcode_dataset(split="train", decontaminate=False)
        clean = load_leetcode_dataset(split="train", decontaminate=True, ngram_size=DEFAULT_NGRAM_SIZE)
        self.assertLess(len(clean), len(raw))


if __name__ == "__main__":
    unittest.main()
