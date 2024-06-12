import argparse

from adapters import PrefixTuningConfig, LoRAConfig, AutoAdapterModel
from transformers import BertTokenizer, Trainer, TrainingArguments

from Logger import logger
from config import DataArgs, Config
from src.dataset import GlueDataset
from utils import count_trainable_parameters, save_results_to_json

def main(dataset_name: str):

    data_args = DataArgs()
    configuration = Config(
        task='prefix+lora',
        dataset=dataset_name
    )
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
    model = AutoAdapterModel.from_pretrained(configuration.MODEL_NAME).eval()
    tokenizer = BertTokenizer.from_pretrained(configuration.MODEL_NAME, do_lower_case=False)
    base_model_parameters = set(model.state_dict().keys())
    for param in model.parameters():
        param.requires_grad = False
    # set up peft
    prefix_config = PrefixTuningConfig(flat=False, prefix_length=30)
    # set up lora config
    lora_config = LoRAConfig(r=8, alpha=16)
    model.add_adapter("lora", config=lora_config, set_active=True)
    model.add_adapter("prefix", config=prefix_config, set_active=True)

    # debugging purpose
    logger.info(f"New Parameters Added by Adapters: {set(model.state_dict().keys()) - base_model_parameters}")

    total_params, trainable_params = count_trainable_parameters(model)
    logger.info(f"Total parameters: {total_params} || Trainable parameters: {trainable_params} ({trainable_params/total_params*100} %)")

    # Initialize the dataset
    glue_dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=dataset_name, training_args=training_args)

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
        'prefix+lora',
        dataset_name, eval_results
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IA3")
    parser.add_argument('dataset', choices=['mnli', 'qnli', 'qqp', 'sst2'], help='Select the dataset to use')
    args = parser.parse_args()
    main(args.dataset)
