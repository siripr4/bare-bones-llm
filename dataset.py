from datasets import load_dataset


def load_and_tokenize_dataset(tokenizer):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512, )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    return tokenized_datasets