import warnings
from os.path import join

from torch import nn
from transformers import Trainer, TrainingArguments, BertForSequenceClassification

from Logger import logger
from config import DataArgs, Config
from src.dataset import GlueDataset
from utils import save_results_to_json, count_trainable_parameters

warnings.simplefilter("ignore")

import torch
from transformers import BertTokenizer


class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


if __name__ == '__main__':
    DATASET = 'mnli'
    # DATASET = 'qnli'
    # DATASET = 'qqp'
    # DATASET = "sst2"

    data_args = DataArgs()
    configuration = Config(task='soft_prompting', dataset=DATASET)
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

    # Initialize the dataset, tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        configuration.MODEL_NAME,
        do_lower_case=False
    )

    glue_dataset = GlueDataset(
        tokenizer,
        data_args=data_args,
        dataset_name=DATASET,
        training_args=training_args
    )

    model = BertForSequenceClassification.from_pretrained(configuration.MODEL_NAME)

    total_params, trainable_params = count_trainable_parameters(model)

    # soft config
    n_tokens = 20
    initialize_from_vocab = True

    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_tokens=n_tokens,
                          initialize_from_vocab=initialize_from_vocab)

    model.set_input_embeddings(s_wte)
    logger.info(f"Total parameters count: {total_params}")
    logger.info(f"Trainable parameters count: {trainable_params} ({trainable_params / total_params * 100}%)")

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
