import logging

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator, BertTokenizer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from config import DataArgs, Config

logger = logging.getLogger(__name__)

TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
}


class GlueDataset:
    def __init__(self, tokenizer: AutoTokenizer, data_args, dataset_name, training_args) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.dataset_name = dataset_name
        # Load the dataset
        raw_datasets = load_dataset("glue", dataset_name)

        # Prepare labels
        self.label_list = raw_datasets["train"].features["label"].names
        self.num_labels = len(self.label_list)
        self.sentence1_key, self.sentence2_key = TASK_TO_KEYS[self.dataset_name]

        # Set padding strategy
        self.padding = "max_length" if self.data_args.pad_to_max_length else False

        # Check max sequence length
        assert self.data_args.max_seq_length < self.tokenizer.model_max_length

        self.max_seq_length = min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

        # Preprocess dataset
        raw_datasets = self._preprocess_datasets(raw_datasets)

        # Split datasets
        self.train_dataset = self._get_dataset(
            raw_datasets,
            "train",
            data_args.max_train_samples
        )
        self.eval_dataset = self._get_dataset(
            raw_datasets,
            "validation_matched" if dataset_name == "mnli" else "validation",
            data_args.max_eval_samples
        )

        self.predict_dataset = self._get_dataset(
            raw_datasets,
            "test_matched" if dataset_name == "mnli" else "test",
            data_args.max_predict_samples
        )

        # Load metric
        self.metric = evaluate.load("glue", self.dataset_name)

        # Set data collator
        self.data_collator = self._get_data_collator()

    def _preprocess_datasets(self, raw_datasets):
        """Preprocess the datasets using the tokenizer."""
        return raw_datasets.map(
            self._preprocess_function,
            batched=True,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    def _preprocess_function(self, examples):
        """Tokenize the texts."""
        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (
                examples[self.sentence1_key], examples[self.sentence2_key])
        )
        return self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

    @staticmethod
    def _get_dataset(raw_datasets, split, max_samples):
        """Get dataset split with optional sample limit."""
        if max_samples is not None:
            return raw_datasets[split].select(range(max_samples))
        return raw_datasets[split]

    def _get_data_collator(self):
        """Determine the data collator."""
        if self.data_args.pad_to_max_length:
            return default_data_collator
        elif self.training_args.fp16:
            return DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        return None

    def compute_metrics(self, p: EvalPrediction):
        """Compute evaluation metrics."""
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return self.metric.compute(predictions=np.argmax(preds, axis=1), references=p.label_ids)


if __name__ == "__main__":



    # Initial
    for dataset_name in list(TASK_TO_KEYS.keys()):
        data_args = DataArgs()
        configuration = Config(
            task='ia3',
            dataset=dataset_name
        )
        # Load the tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(configuration.MODEL_NAME, do_lower_case=False)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=configuration.MODEL_OUTPUT_DIR,
            num_train_epochs=configuration.EPOCHS,
            per_device_train_batch_size=configuration.BATCH_SIZE,
            per_device_eval_batch_size=configuration.BATCH_SIZE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='logs',
            logging_steps=10,

        )
        dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=dataset_name, training_args=training_args)

        print(f"{dataset_name} \t train: {np.unique(dataset.train_dataset['label'])} \t eval: {np.unique(dataset.eval_dataset['label'])}"
              f"\t test: {np.unique(dataset.predict_dataset['label'])}")