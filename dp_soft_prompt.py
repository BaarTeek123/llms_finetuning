import argparse

import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer, TrainingArguments

from Logger import logger
from config import DataArgs
from config import PrivateConfig
from dpsgd import train_prompt_dpsgd
from src.dataset import GlueDataset
from src.soft_prompt_embedding import BertForSequenceClassificationWithSoftPrompt
from utils import _dataset_to_tensordataset, evaluate, save_results_to_json, count_trainable_parameters


def main(dataset_name: str, epsilon: float):
    data_args = DataArgs()

    configuration = PrivateConfig(task='dp soft-prompting', dataset=dataset_name)

    training_args = TrainingArguments(
        output_dir=configuration.MODEL_OUTPUT_DIR,
        num_train_epochs=configuration.EPOCHS,
        per_device_train_batch_size=configuration.BATCH_SIZE,
        per_device_eval_batch_size=configuration.BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=10
    )

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(configuration.MODEL_NAME, do_lower_case=False)
    # Initialize the dataset
    glue_dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=dataset_name, training_args=training_args)

    model = BertForSequenceClassificationWithSoftPrompt(configuration.MODEL_NAME, num_soft_tokens=50,
                                                        num_labels=glue_dataset.num_labels)

    train_dataloader = DataLoader(_dataset_to_tensordataset(glue_dataset.train_dataset),
                                  batch_size=configuration.BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(_dataset_to_tensordataset(glue_dataset.eval_dataset),
                                 sampler=SequentialSampler(_dataset_to_tensordataset(glue_dataset.eval_dataset)),
                                 batch_size=configuration.BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()

    total_params, trainable_params = count_trainable_parameters(model)
    logger.info(
        f"Total parameters: {total_params} || Trainable parameters: {trainable_params} ({trainable_params / total_params * 100}%)")

    model, train_epsilon, delta = train_prompt_dpsgd(
        model, train_dataloader, configuration.EPOCHS, configuration.LR, configuration.MAX_GRAD_NORM,
        epsilon, configuration.DELTA, device,
    )
    logger.info("Training finished")
    logger.info(f"epsilon: {train_epsilon}, delta: {delta}")

    test_eval_loss, test_eval_accuracy = evaluate(model, eval_dataloader, device)
    eval_results = {
        "loss": test_eval_loss,
        "accuracy": test_eval_accuracy
    }
    logger.info("Evaluation finished")
    logger.info(f"Evaluation results: {eval_results}.")


    save_results_to_json(
        configuration.RESULTS_PATH,
        'prompt_dpsgd',
        dataset_name,
        train_results={},
        eval_results=eval_results,
        additional_comments={
            "trainable parameters": trainable_params,
            "total parameters": total_params,
            "trainable parameters ratio (%)": trainable_params / total_params * 100,
            "configuration": configuration.model_dump_json(),
            "epsilon": train_epsilon
        }
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run prompt_dpsgd")
    parser.add_argument('dataset', choices=['mnli', 'qnli', 'qqp', 'sst2'], help='Select the dataset to use')
    parser.add_argument('epsilon', type=lambda x: (
        float(x) if float(x) > 0 else argparse.ArgumentTypeError(f"{x} is not a positive float or int")),
                        help='Epsilon value for DP (must be > 0)')

    args = parser.parse_args()
    try:
        main(args.dataset, args.epsilon)
    except Exception as ex:
        logger.error(f"Something went wrong {ex}")
