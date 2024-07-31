# fine_tune_model.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

def fine_tune_model(df, model_name='all-MiniLM-L6-v2', epochs=3, batch_size=32):
    # Load a pre-trained model
    model = SentenceTransformer(model_name)
    
    # Prepare training data
    train_examples = [InputExample(texts=[row['Question'], row['Answer']]) for _, row in df.iterrows()]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Fine-tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100)
    
    # Save the model
    model.save('fine_tuned_model')
    print("Model fine-tuning completed and saved as 'fine_tuned_model'.")
    
if __name__ == "__main__":
    df = pd.read_csv('cleaned_FAQ.csv')
    fine_tune_model(df)
