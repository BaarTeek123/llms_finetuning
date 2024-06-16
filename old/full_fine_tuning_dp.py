import argparse

from dp_transformers import PrivacyArguments
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from Logger import logger
from config import DataArgs, PrivateConfig
from src.dataset import GlueDataset
from utils import count_trainable_parameters, save_results_to_json

from opacus import PrivacyEngine
import dp_transformers




def main(dataset_name: str):
    data_args = DataArgs()
    privacy_engine = PrivacyEngine()
    configuration = PrivateConfig(task='full_fine_tuning_dp', dataset=dataset_name)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=configuration.MODEL_OUTPUT_DIR,
        num_train_epochs=configuration.EPOCHS,
        per_device_train_batch_size=configuration.BATCH_SIZE,
        per_device_eval_batch_size=configuration.BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='../logs',
        logging_steps=10
    )

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(configuration.MODEL_NAME, do_lower_case=False)
    # Initialize the dataset
    glue_dataset = GlueDataset(tokenizer, data_args=data_args, dataset_name=dataset_name, training_args=training_args)

    model = BertForSequenceClassification.from_pretrained(configuration.MODEL_NAME,
                                                          num_labels=glue_dataset.num_labels)

    total_params, trainable_params = count_trainable_parameters(model)

    logger.info(f"Total parameters count: {total_params}")
    logger.info(f"Trainable parameters count: {trainable_params} ({trainable_params / total_params * 100}%)")
    # set model to train mode
    model.train()

    optimizer = SGD(model.parameters(), lr=configuration.LR)

    privacy_args = PrivacyArguments(
        per_sample_max_grad_norm=configuration.MAX_GRAD_NORM,
        target_epsilon=configuration.EPSILON,
        target_delta=configuration.DELTA
    )

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=glue_dataset.train_dataset,
        eval_dataset=glue_dataset.eval_dataset,
        privacy_args=privacy_args,
        optimizers=(optimizer, StepLR(optimizer, step_size=30, gamma=0.1))
    )


    train_results = trainer.train()
    # logger.info("Train results: %s", train_results.metrics)
    # trainer.log_metrics("eval", train_results.metrics)
    # trainer.save_metrics("eval", train_results.metrics)
    # trainer.save_state()
    # #
    # # # set to evaluation mode
    # # model.eval()
    # #
    # # # evaluate the model
    # # eval_results = trainer.evaluate()
    # # logger.info("Evaluation results: %s", eval_results)
    # # trainer.log_metrics("eval", eval_results)
    # # trainer.save_metrics("eval", eval_results)
    # #
    # # # Save evaluation results to JSON
    # # save_results_to_json(
    # #     configuration.RESULTS_PATH,
    # #     'full fine-tuning',
    # #     dataset_name,
    # #     train_results=train_results,
    # #     eval_results=eval_results,
    # #     additional_comments={
    # #         "trainable parameters": trainable_params,
    # #         "total parameters": total_params,
    # #         "trainable parameters ratio (%)": trainable_params / total_params * 100
    # #     }
    # # )


    # for epoch in range(1, configuration.EPOCHS + 1):
    #     losses = []
    #
    #     with BatchMemoryManager(
    #             data_loader=train_dataloader,
    #             max_physical_batch_size=configuration.MAX_PHYSICAL_BATCH_SIZE,
    #             optimizer=optimizer
    #     ) as memory_safe_data_loader:
    #         for step, batch in enumerate(tqdm(memory_safe_data_loader)):
    #             optimizer.zero_grad()
    #
    #             batch = tuple(t.to(device) for t in batch)
    #             inputs = {'input_ids': batch[0],
    #                       'attention_mask': batch[1],
    #                       'token_type_ids': batch[2],
    #                       'labels': batch[3]}
    #
    #             outputs = model(**inputs)  # output = loss, logits, hidden_states, attentions
    #
    #             loss = outputs[0]
    #             loss.backward()
    #             losses.append(loss.item())
    #
    #             optimizer.step()
    #
    #             if step > 0 and step % LOGGING_INTERVAL == 0:
    #                 train_loss = np.mean(losses)
    #                 eps = privacy_engine.get_epsilon(DELTA)
    #
    #                 eval_loss, eval_accuracy = evaluate(model)
    #
    #                 print(
    #                     f"Epoch: {epoch} | "
    #                     f"Step: {step} | "
    #                     f"Train loss: {train_loss:.3f} | "
    #                     f"Eval loss: {eval_loss:.3f} | "
    #                     f"Eval accuracy: {eval_accuracy:.3f} | "
    #                     f"ɛ: {eps:.2f}"
    #                 )
    #
    # # define evaluation cycle
    # def evaluate(model, test_dataloader,device):
    #     model.eval()
    #
    #     loss_arr = []
    #     accuracy_arr = []
    #
    #     for batch in test_dataloader:
    #         batch = tuple(t.to(device) for t in batch)
    #
    #         with torch.no_grad():
    #             inputs = {'input_ids': batch[0],
    #                       'attention_mask': batch[1],
    #                       'token_type_ids': batch[2],
    #                       'labels': batch[3]}
    #
    #             outputs = model(**inputs)
    #             loss, logits = outputs[:2]
    #
    #             preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
    #             labels = inputs['labels'].detach().cpu().numpy()
    #
    #             loss_arr.append(loss.item())
    #             accuracy_arr.append(accuracy(preds, labels))
    #     model.train()
    #     return np.mean(loss_arr), np.mean(accuracy_arr)
    #
    # def accuracy(preds, labels):
    #     return (preds == labels).mean()
    # # # Train the model


if __name__ == '__main__':
    main('qnli')