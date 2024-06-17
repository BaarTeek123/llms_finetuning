# import warnings
#
# from peft import get_peft_model
# from torch import nn
# from torch.optim import SGD
# from transformers import BertForSequenceClassification
#
# from Logger import logger
#
# warnings.simplefilter("ignore")
#
# import torch
#
#
# class SoftPrompt(nn.Module):
#     def __init__(self, num_tokens: int, embedding_size: int):
#         super(SoftPrompt, self).__init__()
#         self.soft_prompts = nn.Parameter(torch.randn(num_tokens, embedding_size))
#
#     def forward(self, inputs_embeds):
#         batch_size = inputs_embeds.size(0)
#         soft_prompts_expanded = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
#         inputs_embeds = torch.cat([soft_prompts_expanded, inputs_embeds], dim=1)
#         return inputs_embeds
#
#
# class BertForSequenceClassificationWithSoftPrompt(nn.Module):
#     def __init__(self, model_name, num_soft_tokens, **kwargs):
#         super(BertForSequenceClassificationWithSoftPrompt, self).__init__()
#         self.bert = BertForSequenceClassification.from_pretrained(model_name, **kwargs)
#
#         # freeze BERT
#         for param in self.bert.parameters():
#             param.requires_grad = False
#         self.embedding_size = self.bert.config.hidden_size
#         self.soft_prompt = SoftPrompt(num_soft_tokens, self.embedding_size)
#         self.num_soft_prompts = num_soft_tokens
#
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         # Get input embeddings
#         inputs_embeds = self.bert.bert.embeddings(input_ids)
#
#         # Debugging print statement
#         logger.info(f"Input embeddings size: {inputs_embeds.size()}")
#
#         # Add soft prompts
#         inputs_embeds = self.soft_prompt(inputs_embeds)
#
#         # Debugging print statement
#         logger.info(f"Inputs embeds after soft prompt size: {inputs_embeds.size()}")
#
#         # Adjust attention mask
#         attention_mask = torch.cat(
#             [torch.ones((attention_mask.size(0), self.num_soft_prompts), device=attention_mask.device), attention_mask],
#             dim=1)
#
#         # Debugging print statement
#         logger.info(f"Attention mask size after concat: {attention_mask.size()}")
#
#         outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
#         return outputs
#
#
# class BertForSequenceClassificationWithSoftPromptPeft(nn.Module):
#     def __init__(self, model_name, num_soft_tokens, peft_config, **kwargs):
#         super(BertForSequenceClassificationWithSoftPromptPeft, self).__init__()
#         self.bert = BertForSequenceClassification.from_pretrained(model_name, return_dict=True, **kwargs)
#
#         self.bert = get_peft_model(self.bert, peft_config)
#
#         self.embedding_size = self.bert.config.hidden_size
#         self.soft_prompt = SoftPrompt(num_soft_tokens, self.embedding_size)
#         self.num_soft_prompts = num_soft_tokens
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         inputs_embeds = self.bert.bert.embeddings(input_ids)
#         inputs_embeds = self.soft_prompt(inputs_embeds)
#         attention_mask = torch.cat(
#             [torch.ones((attention_mask.size(0), self.num_soft_prompts), device=attention_mask.device), attention_mask],
#             dim=1)
#         outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
#         return outputs
#
#
# # Helper functions
# def clip_gradients(grad, max_norm):
#     norm = grad.norm(2)
#     scale = max_norm / (norm + 1e-6)
#     return grad if scale >= 1 else grad * scale
#
#
# def add_noise(grad, noise_scale):
#     noise = torch.randn_like(grad) * noise_scale
#     return grad + noise
#
#
# def compute_privacy_cost(T, batch_size, delta, noise_scale, dataset):
#     epsilon = T * (batch_size / len(dataset)) * noise_scale
#     return epsilon, delta
#
#
#
# def train_prompt_dpsgd(model, dataloader, num_epochs, learning_rate, max_norm, noise_scale, delta, device):
#     optimizer = SGD([model.soft_prompt.soft_prompts], lr=learning_rate)
#     T = num_epochs * len(dataloader)  # Total number of iterations
#
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             batch = tuple(t.to(device) for t in batch)
#             inputs = {'input_ids': batch[0],
#                       'attention_mask': batch[1],
#                       'token_type_ids': batch[2],
#                       'labels': batch[3]}
#
#             optimizer.zero_grad()
#
#             # Forward pass
#             outputs = model(**inputs)
#             loss = outputs.loss
#
#             # Compute gradients
#             loss.backward()
#
#             # Clip and add noise to gradients
#             for param in model.soft_prompt.parameters():
#                 param.grad = clip_gradients(param.grad, max_norm)
#                 param.grad = add_noise(param.grad, noise_scale)
#
#             # Update parameters
#             optimizer.step()
#
#     # Compute overall privacy cost
#     epsilon, delta = compute_privacy_cost(T, dataloader.batch_size, delta, noise_scale)
#     return epsilon, delta
#
#


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np


class SoftPrompt(nn.Module):
    def __init__(self, num_tokens: int, embedding_size: int):
        super(SoftPrompt, self).__init__()
        self.soft_prompts = nn.Parameter(torch.randn(num_tokens, embedding_size))

    def forward(self, inputs_embeds):
        batch_size = inputs_embeds.size(0)
        soft_prompts_expanded = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([soft_prompts_expanded, inputs_embeds], dim=1)
        return inputs_embeds


class BertForSequenceClassificationWithSoftPrompt(nn.Module):
    def __init__(self, model_name, num_soft_tokens, **kwargs):
        super(BertForSequenceClassificationWithSoftPrompt, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, **kwargs)

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.embedding_size = self.bert.config.hidden_size
        self.soft_prompt = SoftPrompt(num_soft_tokens, self.embedding_size)
        self.num_soft_prompts = num_soft_tokens

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Get input embeddings
        inputs_embeds = self.bert.bert.embeddings(input_ids)

        # Add soft prompts
        inputs_embeds = self.soft_prompt(inputs_embeds)

        # Adjust attention mask
        attention_mask = torch.cat(
            [torch.ones((attention_mask.size(0), self.num_soft_prompts), device=attention_mask.device), attention_mask],
            dim=1
        )

        # Adjust token type ids
        if token_type_ids is not None:
            token_type_ids = torch.cat(
                [torch.zeros((token_type_ids.size(0), self.num_soft_prompts), device=token_type_ids.device,
                             dtype=token_type_ids.dtype), token_type_ids],
                dim=1
            )

        # Pass the modified embeddings and attention mask to the BERT model
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs
        )
        return outputs
