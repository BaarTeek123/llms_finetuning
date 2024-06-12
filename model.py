import warnings
from os.path import join

from transformers import Trainer, TrainingArguments

from Logger import logger
from config import DataArgs, Config
from dataset import GlueDataset
from utils import save_results_to_json, count_trainable_parameters

warnings.simplefilter("ignore")

import torch
from transformers import AutoModel, AutoConfig, BertTokenizer
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


if __name__ == '__main__':
    DATASET = 'mnli'
    # DATASET = 'qnli'
    # DATASET = 'qqp'
    # DATASET = "sst2"

    data_args = DataArgs()
    configuration = Config(task='additional_layer')
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
    tokenizer = BertTokenizer.from_pretrained(configuration.MODEL_NAME, do_lower_case=False)

    glue_dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=DATASET, training_args=training_args)

    model = NoTinyBERT(
        configuration.MODEL_NAME,
        num_labels=glue_dataset.num_labels
    )

    total_params, trainable_params = count_trainable_parameters(model)

    logger.info(f"Total parameters: {total_params} || Trainable parameters: {trainable_params} ({trainable_params/total_params*100}%)")

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
    save_results_to_json(join('out', 'evaluation_results.json'), 'NoTinyBERT', DATASET, eval_results)
