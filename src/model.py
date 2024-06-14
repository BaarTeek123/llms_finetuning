import warnings

warnings.simplefilter("ignore")

import torch
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput


class NoTinyBERT(torch.nn.Module):
    def __init__(self, checkpoint, num_labels, dropout=torch.nn.Dropout(0.1)):
        super(NoTinyBERT, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        config = AutoConfig.from_pretrained(
            checkpoint,
            output_attentions=True,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(checkpoint, config=config)
        # Freeze the pre-trained model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.dropout = dropout
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Add custom layers
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output[:, 0, :])

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

