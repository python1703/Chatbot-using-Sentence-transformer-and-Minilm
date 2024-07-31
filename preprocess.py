# preprocess.py
import pandas as pd

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Data cleaning
    df.dropna(inplace=True)  # Remove rows with missing values
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    
    # Further cleaning 
    df['Question'] = df['Question'].str.lower().str.replace('[^\w\s]', '')
    df['Answer'] = df['Answer'].str.lower().str.replace('[^\w\s]', '')
    
    return df

if __name__ == "__main__":
    df = preprocess_data('FAQ.csv')
    df.to_csv('cleaned_FAQ.csv', index=False)
    print("Data preprocessing completed and saved to cleaned_FAQ.csv.")
