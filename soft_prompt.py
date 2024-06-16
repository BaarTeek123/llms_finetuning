import argparse

from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer

from Logger import logger
from config import DataArgs, Config
from src.dataset import GlueDataset
from src.soft_prompt_embedding import BertForSequenceClassificationWithSoftPrompt
from utils import save_results_to_json, count_trainable_parameters


def main(dataset_name):
    data_args = DataArgs()
    configuration = Config(task='soft_prompting', dataset=dataset_name)

    # debugging purpose
    base_model_parameters = set(
        BertForSequenceClassification.from_pretrained(configuration.MODEL_NAME).state_dict().keys()
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=configuration.MODEL_OUTPUT_DIR,
        num_train_epochs=configuration.EPOCHS,
        per_device_train_batch_size=configuration.BATCH_SIZE,
        per_device_eval_batch_size=configuration.BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='../logs',
        logging_steps=10,
    )

    # Initialize the dataset, tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        configuration.MODEL_NAME,
        do_lower_case=False
    )

    glue_dataset = GlueDataset(
        tokenizer,
        data_args=data_args,
        dataset_name=dataset_name,
        training_args=training_args
    )

    model = BertForSequenceClassificationWithSoftPrompt(configuration.MODEL_NAME, num_soft_tokens=50,
                                                        num_labels=glue_dataset.num_labels)
    # debugging purpose
    logger.info(f"New Parameters Added by Adapters: {set(model.state_dict().keys()) - base_model_parameters}")

    total_params, trainable_params = count_trainable_parameters(model)
    logger.info(f"Total parameters count: {total_params}")
    logger.info(f"Trainable parameters count: {trainable_params} ({trainable_params / total_params * 100}%)")
    logger.info(f"Unique labels: {set(glue_dataset.train_dataset['label'])}")
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
    train_results = trainer.train()
    logger.info("Train results: %s", train_results.metrics)
    trainer.log_metrics("eval", train_results.metrics)
    trainer.save_metrics("eval", train_results.metrics)
    trainer.save_state()

    # set to evaluation mode
    model.eval()

    # evaluate the model
    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    # Save evaluation results to JSON
    save_results_to_json(
        configuration.RESULTS_PATH,
        'soft prompt',
        dataset_name,
        train_results=train_results,
        eval_results=eval_results,
        additional_comments={
            "trainable parameters": trainable_params,
            "total parameters": total_params,
            "trainable parameters ratio (%)": trainable_params / total_params * 100
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Soft prompt")
    parser.add_argument('dataset', choices=['mnli', 'qnli', 'qqp', 'sst2'], help='Select the dataset to use')
    args = parser.parse_args()
    try:
        main(args.dataset)
    except Exception as ex:
        logger.error(f"Something went wrong {ex}")
