
import argparse

from config import get_training_arguments
from dataset import load_and_tokenize_dataset
from model import get_pretrained_model
from train import train_model
from utils import generate_text, save_model

def main(model_name, output_dir, num_train_epochs, batch_size, input_text):
    model, tokenizer = get_pretrained_model(model_name)
    dataset = load_and_tokenize_dataset(tokenizer)
    training_args = get_training_arguments(output_dir=output_dir, num_train_epochs=num_train_epochs, batch_size=batch_size)

    train_model(model, dataset, training_args)
    save_model(model, tokenizer, './fine_tuned_model')

    if input_text:
        output_text = generate_text(model, tokenizer, input_text)
        print(f"Generated Text: {output_text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a pre-trained language model.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="The name of the pre-trained model to use.")
    parser.add_argument("--output_dir", type=str, default="./results", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device during training.")
    parser.add_argument("--input_text", type=str, default="", help="Input text for generating text after training.")
    
    args = parser.parse_args()
    main(args.model_name, args.output_dir, args.num_train_epochs, args.batch_size, args.input_text)
