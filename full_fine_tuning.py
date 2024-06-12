from os.path import join

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from Logger import logger
from config import DataArgs, Config
from dataset import GlueDataset
from utils import count_trainable_parameters, save_results_to_json

if __name__ == '__main__':
    DATASET = 'mnli'
    DATASET = 'qnli'
    # DATASET = 'qqp'
    # DATASET = "sst2"

    data_args = DataArgs()
    configuration = Config()
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=configuration.OUTPUT_DIR,
        num_train_epochs=configuration.EPOCHS,
        per_device_train_batch_size=configuration.BATCH_SIZE,
        per_device_eval_batch_size=configuration.BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,

    )

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(configuration.MODEL_NAME, do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained(configuration.MODEL_NAME)

    total_params, trainable_params = count_trainable_parameters(model)

    logger.info(f"Total parameters count: {total_params}")
    logger.info(f"Trainable parameters count: {trainable_params} ({trainable_params/total_params*100}%)")

    # Initialize the dataset
    glue_dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=DATASET, training_args=training_args)

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
    save_results_to_json(join('out', 'evaluation_results.json'), 'TinyBERT', DATASET, eval_results)