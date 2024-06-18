from transformers import Trainer, TrainingArguments

def train_model(model, dataset, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()