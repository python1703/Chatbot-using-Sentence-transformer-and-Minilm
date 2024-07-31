from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Custom Dataset class for leave data
class LeaveDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        leave_text = self.data[idx]

        input_text = f"{leave_text}"
        inputs = self.tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': inputs['input_ids'].squeeze(0)  # Use the same input as label
        }

# Fine-tune function for leave data
def fine_tune_leave_model(data, model_name='gpt2', epochs=3, batch_size=4, lr=1e-5):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare data
    dataset = LeaveDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Fine-tune the model
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(dataloader):.4f}")

    # Save the fine-tuned model
    model.save_pretrained('fine_tuned_leave_model')
    tokenizer.save_pretrained('fine_tuned_leave_model')
    print("Model fine-tuning completed and saved as 'fine_tuned_leave_model'.")

if __name__ == "__main__":
    # Load leave data from leave.txt
    with open('leave.txt', 'r', encoding='utf-8') as file:
        leave_data = file.readlines()

    # Fine-tune the model on leave data
    fine_tune_leave_model(leave_data, epochs=3, batch_size=4, lr=1e-5)
