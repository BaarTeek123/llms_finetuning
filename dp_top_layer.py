import numpy as np
import torch
import argparse

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


from torch.utils.data import DataLoader, SequentialSampler
from Logger import logger
from config import DataArgs, PrivateConfig
from src.dataset import GlueDataset
from src.model import NoTinyBERT
from utils import count_trainable_parameters, save_results_to_json, train_model, evaluate, _dataset_to_tensordataset
from config import PrivateConfig
from opacus import PrivacyEngine




def main(dataset_name: str, epsilon: float):
    data_args = DataArgs()
    privacy_engine = PrivacyEngine()
    configuration = PrivateConfig(task='full_fine_tuning_dp', dataset=dataset_name)

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

    model = NoTinyBERT(
        configuration.MODEL_NAME,
        num_labels=glue_dataset.num_labels
    )

    train_dataloader = DataLoader(_dataset_to_tensordataset(glue_dataset.train_dataset),
                                  batch_size=configuration.BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(_dataset_to_tensordataset(glue_dataset.eval_dataset),
                                 sampler=SequentialSampler(_dataset_to_tensordataset(glue_dataset.eval_dataset)),
                                 batch_size=configuration.BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()
    # Define optimizer
    optimizer = SGD(model.parameters(), lr=configuration.LR)

    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_delta=configuration.DELTA,
        target_epsilon=epsilon,
        epochs=configuration.EPOCHS,
        max_grad_norm=configuration.MAX_GRAD_NORM,
    )

    total_params, trainable_params = count_trainable_parameters(model)
    logger.info(
        f"Total parameters: {total_params} || Trainable parameters: {trainable_params} ({trainable_params / total_params * 100}%)")

    model, train_results = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=eval_dataloader,
        optimizer=optimizer,
        epochs=configuration.EPOCHS,
        privacy_engine=privacy_engine,
        delta=configuration.DELTA,
        device=device
    )

    test_eval_loss, test_eval_accuracy = evaluate(model, eval_dataloader, device)
    eval_results = {
        "loss": test_eval_loss,
        "accuracy": test_eval_accuracy
    }

    save_results_to_json(
        configuration.RESULTS_PATH,
        'NoTinyBERT',
        dataset_name,
        train_results=train_results,
        eval_results=eval_results,
        additional_comments={
            "trainable parameters": trainable_params,
            "total parameters": total_params,
            "trainable parameters ratio (%)": trainable_params / total_params * 100,
            "configuration": configuration.model_dump()
        }
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NoTinyBERT")
    parser.add_argument('dataset', choices=['mnli', 'qnli', 'qqp', 'sst2'], help='Select the dataset to use')
    parser.add_argument('epsilon', type=lambda x: (
        float(x) if float(x) > 0 else argparse.ArgumentTypeError(f"{x} is not a positive float or int")),
                        help='Epsilon value for DP (must be > 0)', default=np.inf)

    args = parser.parse_args()
    try:
        main(args.dataset, args.epsilon)
    except Exception as ex:
        logger.error(f"Something went wrong {ex}")
