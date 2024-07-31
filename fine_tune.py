import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tokenize_function(examples, tokenizer):
    inputs = tokenizer(examples['Question'], padding="max_length", truncation=True, max_length=256)
    targets = tokenizer(examples['Answer'], padding="max_length", truncation=True, max_length=256)
    inputs['labels'] = targets['input_ids']

    for key in inputs:
        if isinstance(inputs[key], list) and all(isinstance(i, int) for i in inputs[key]):
            if max(inputs[key]) >= tokenizer.vocab_size:
                logger.warning(f"Out of range index found in {key}: {max(inputs[key])}")
    return inputs

if __name__ == "__main__":
    model_name = "openchat/openchat_3.5"
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
    
    # Resize the model embedding to accommodate the new tokens
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load the prepared dataset
    dataset = load_from_disk('prepared_dataset')
    
    # Split the dataset into training and evaluation sets
    split_dataset = dataset.train_test_split(test_size=0.1)  # 10% for evaluation
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    # Tokenize the datasets
    tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True, remove_columns=['Question', 'Answer'])
    tokenized_eval_dataset = eval_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True, remove_columns=['Question', 'Answer'])
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_total_limit=2,
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset  # Use the evaluation dataset
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    model_save_path = "fine-tuned-model"  # Replace with your desired path
    tokenizer_save_path = "fine-tuned-tokenizer"  # Replace with your desired path
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
