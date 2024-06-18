from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_pretrained_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer