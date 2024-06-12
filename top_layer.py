import argparse

from transformers import Trainer, TrainingArguments, BertTokenizer

from Logger import logger
from config import DataArgs, Config
from src.dataset import GlueDataset
from model import NoTinyBERT
from utils import save_results_to_json, count_trainable_parameters


def main(dataset_name: str):
    data_args = DataArgs()
    configuration = Config(task='additional_layer', dataset=dataset_name)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=configuration.MODEL_OUTPUT_DIR,
        num_train_epochs=configuration.EPOCHS,
        per_device_train_batch_size=configuration.BATCH_SIZE,
        per_device_eval_batch_size=configuration.BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize the dataset, tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(configuration.MODEL_NAME, do_lower_case=False)

    glue_dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=dataset_name, training_args=training_args)

    model = NoTinyBERT(
        configuration.MODEL_NAME,
        num_labels=glue_dataset.num_labels
    )

    total_params, trainable_params = count_trainable_parameters(model)

    logger.info(
        f"Total parameters: {total_params} || Trainable parameters: {trainable_params} ({trainable_params / total_params * 100}%)")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=glue_dataset.train_dataset,
        eval_dataset=glue_dataset.eval_dataset,
        tokenizer=tokenizer,
        data_collator=glue_dataset.data_collator,
        compute_metrics=glue_dataset.compute_metrics
    )

    # Train the model
    trainer.train()

    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    # Save evaluation results to JSON
    save_results_to_json(
        configuration.RESULTS_PATH,
        'NoTinyBERT',
        dataset_name, eval_results
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NoTinyBERT")
    parser.add_argument('dataset', choices=['mnli', 'qnli', 'qqp', 'sst2'], help='Select the dataset to use')
    args = parser.parse_args()
    main(args.dataset)
