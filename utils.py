from Logger import logger
from json import load, dump
import torch
from tqdm import tqdm
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
from torch.utils.data import TensorDataset

import json
from pathlib import Path
from typing import Union

import pandas as pd


def create_output_file(file_path_to_json: Union[str, Path]):
    data = json.load(open(file_path_to_json))
    df_data = {
        "Dataset": [],
        "Accuracy": [],
        "Task name": []
    }

    for k, v in data.items():
        if k == 'Comments': continue
        if 'task_type' in list(v.keys()):
            df_data["Task name"].append(v['task_type'])

        if 'dataset_name' in list(v.keys()):
            df_data["Dataset"].append(v['dataset_name'])

        if 'eval results' in list(v.keys()):
            df_data["Accuracy"].append(v['eval results']['eval_accuracy'])

    return pd.DataFrame(df_data).pivot(index="Dataset", columns="Task name", values="Accuracy")


def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _dataset_to_tensordataset(dataset):
    """Convert a Hugging Face dataset to a TensorDataset."""
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    return TensorDataset(dataset['input_ids'], dataset['attention_mask'], dataset['token_type_ids'], dataset['label'])


def save_results_to_json(file_path, task_type, dataset_name, train_results, eval_results,
                         additional_comments: dict = None):
    """Save the evaluation results to a JSON file."""

    try:
        with open(file_path, 'r') as file:
            data = load(file)
    except FileNotFoundError:
        data = {}

    # Update the results for the specified model and dataset
    key = f"{task_type}_{dataset_name}"
    data[key] = {
        "task_type": task_type,
        "dataset_name": dataset_name,
        "train results": train_results,
        "eval results": eval_results
    }
    data[key].update({"Comments": additional_comments})

    with open(file_path, 'w') as file:
        dump(data, file, indent=4)

    logger.info(f"Results saved to {file_path}")


def accuracy(preds, labels):
    return (preds == labels).mean()


def train_model(model, optimizer, train_dataloader, test_dataloader, device, privacy_engine=None, epochs=1, delta=1e-5,
                max_grad_norm=1.0):
    train_results_list = []

    for epoch in range(1, epochs + 1):
        losses = []

        with BatchMemoryManager(
                data_loader=train_dataloader,
                max_physical_batch_size=max_grad_norm,
                optimizer=optimizer
        ) as memory_safe_data_loader:
            for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                optimizer.zero_grad()

                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}

                outputs = model(**inputs)

                loss = outputs[0]
                loss.backward()

                losses.append(loss.item())

                optimizer.step()

                if step > 0 and step % 5000 == 0:
                    train_loss = np.mean(losses)
                    eps = privacy_engine.get_epsilon(delta) if privacy_engine else None

                    eval_loss, eval_accuracy = evaluate(model,
                                                        test_dataloader,
                                                        device)

                    logger.info(
                        f"Epoch: {epoch} | "
                        f"Step: {step} | "
                        f"Train loss: {train_loss:.3f} | "
                        f"Eval loss: {eval_loss:.3f} | "
                        f"Eval accuracy: {eval_accuracy:.3f} | "
                        f"ɛ: {eps:.2f}"
                    )
        train_results_list.append(np.mean(losses))


    return model, train_results_list


def evaluate(model, test_dataloader, device):

    loss_arr = []
    accuracy_arr = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            outputs = model(**inputs)
            loss, logits = outputs[:2]

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs['labels'].detach().cpu().numpy()

            loss_arr.append(loss.item())
            accuracy_arr.append(accuracy(preds, labels))

    return np.mean(loss_arr), np.mean(accuracy_arr)


if __name__ == '__main__':
    # Test the function
    df = create_output_file('results/evaluation_results.json')
    df.to_csv('results/evaluation_results.csv')
    print(df)