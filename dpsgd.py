import torch
from torch.optim import SGD


def clip_gradients(grad, max_norm):
    norm = grad.norm(2)
    scale = max_norm / (norm + 1e-6)
    return grad if scale >= 1 else grad * scale


def add_noise(grad, noise_scale):
    noise = torch.randn_like(grad) * noise_scale
    return grad + noise


def compute_privacy_cost(T, batch_size, delta, noise_scale, dataset):
    epsilon = T * (batch_size / len(dataset)) * noise_scale
    return epsilon, delta


def train_prompt_dpsgd(model, dataloader, num_epochs, learning_rate, max_norm, noise_scale, delta, device):
    optimizer = SGD([model.soft_prompt.soft_prompts], lr=learning_rate)
    T = num_epochs * len(dataloader)  # Total number of iterations

    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3]
            }

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids'),
                labels=inputs['labels']
            )
            loss = outputs.loss

            # Compute gradients
            loss.backward()

            # Clip and add noise to gradients
            for param in model.soft_prompt.parameters():
                param.grad = clip_gradients(param.grad, max_norm)
                param.grad = add_noise(param.grad, noise_scale)

            # Update parameters
            optimizer.step()

    # Compute overall privacy cost
    epsilon, delta = compute_privacy_cost(T, dataloader.batch_size, delta, noise_scale, dataloader.dataset)
    return model, epsilon, delta

