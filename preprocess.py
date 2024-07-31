import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Basic text cleaning
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with a single space
    text = text.lower()  # Convert to lowercase

    # Tokenize the text into sentences
    nltk.download('punkt')
    sentences = sent_tokenize(text)

    # Remove any sentences that are URLs or irrelevant
    sentences = [sentence for sentence in sentences if not re.match(r'http\S+', sentence)]

    # Chunking the data
    chunk_size = 500  # Define the chunk size
    chunked_sentences = [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]

    return chunked_sentences

if __name__ == "__main__":
    chunked_data = preprocess_data('leave.txt')
    with open('cleaned_leave_data.txt', 'w', encoding='utf-8') as f:
        for chunk in chunked_data:
            f.write('\n\n'.join(chunk))
            f.write('\n\n')
    print(f"Data preprocessing completed and saved to cleaned_leave_data.txt ({len(chunked_data)} chunks).")
