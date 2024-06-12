from Logger import logger
from json import load, dump


def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# NotADirectoryError

def save_results_to_json(file_path, task_type, dataset_name, results):
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
        "results": results
    }

    with open(file_path, 'w') as file:
        dump(data, file, indent=4)

    logger.info(f"Results saved to {file_path}")