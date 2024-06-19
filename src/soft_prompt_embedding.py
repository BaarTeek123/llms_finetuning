import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import BertForSequenceClassification


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


class BertForSequenceClassificationWithSoftPromptPeft(nn.Module):
    def __init__(self, model_name, num_soft_tokens, peft_config, **kwargs):
        super(BertForSequenceClassificationWithSoftPromptPeft, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, return_dict=True, **kwargs)

        self.bert = get_peft_model(self.bert, peft_config)

        self.embedding_size = self.bert.config.hidden_size
        self.soft_prompt = SoftPrompt(num_soft_tokens, self.embedding_size)
        self.num_soft_prompts = num_soft_tokens

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        inputs_embeds = self.bert.bert.embeddings(input_ids)
        inputs_embeds = self.soft_prompt(inputs_embeds)
        attention_mask = torch.cat(
            [torch.ones((attention_mask.size(0), self.num_soft_prompts), device=attention_mask.device), attention_mask],
            dim=1)
        # Adjust token type ids
        if token_type_ids is not None:
            token_type_ids = torch.cat(
                [torch.zeros((token_type_ids.size(0), self.num_soft_prompts), device=token_type_ids.device,
                             dtype=token_type_ids.dtype), token_type_ids], dim=1
            )

        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        return outputs