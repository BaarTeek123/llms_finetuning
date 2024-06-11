from os.path import join

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from Logger import logger
from dataset import GlueDataset
from utils import count_trainable_parameters, save_results_to_json
from pydantic import BaseModel

if __name__ == '__main__':
    class DataArgs(BaseModel):
        pad_to_max_length: bool = True
        max_seq_length: int = 128
        overwrite_cache: bool = False
        max_train_samples: int = None
        max_predict_samples: int = None
        max_eval_samples: int = None

        class Config:
            allow_mutation = False


            # max_train_samples = 10000
        # max_predict_samples = 500
        # max_eval_samples = 500


    data_args = DataArgs()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,

    )
    MODEL_NAME = "prajjwal1/bert-tiny"
    DATASET = "sst2"
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

    total_params, trainable_params = count_trainable_parameters(model)

    logger.info(f"Total parameters count: {total_params}")
    logger.info(f"Trainable parameters count: {trainable_params} ({trainable_params/total_params*100}%)")

    # Initialize the dataset
    dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=DATASET, training_args=training_args)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.eval_dataset,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        compute_metrics=dataset.compute_metrics
    )

    # Train the model
    trainer.train()

    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    # Save evaluation results to JSON
    save_results_to_json(join('out', 'evaluation_results.json'), 'TinyBERT', DATASET, eval_results)