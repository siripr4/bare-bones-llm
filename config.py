from transformers import TrainingArguments

def get_training_arguments(output_dir, num_train_epochs, batch_size):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
    )