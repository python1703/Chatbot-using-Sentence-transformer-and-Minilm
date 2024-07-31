import pandas as pd
from datasets import Dataset

def prepare_data(csv_path):
    # Load your CSV data
    df = pd.read_csv(csv_path)
    
    # Ensure the columns are named correctly
    df.columns = ['Question', 'Answer']
    
    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset

if __name__ == "__main__":
    csv_path = 'leave.csv'  # Replace with the path to your CSV file
    dataset = prepare_data(csv_path)
    dataset.save_to_disk('prepared_dataset')  # Save the dataset for later use
