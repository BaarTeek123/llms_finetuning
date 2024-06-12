from os.path import join

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from Logger import logger
from config import DataArgs, Config
from dataset import GlueDataset
from utils import count_trainable_parameters, save_results_to_json

if __name__ == '__main__':
    DATASET = 'mnli'
    # DATASET = 'qnli'
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

    # Initialize the dataset, tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        configuration.MODEL_NAME,
        do_lower_case=False
    )

    glue_dataset = GlueDataset(
        tokenizer,
        data_args=data_args,
        dataset_name=DATASET,
        training_args=training_args
    )

    model = BertForSequenceClassification.from_pretrained(
        configuration.MODEL_NAME
    )


