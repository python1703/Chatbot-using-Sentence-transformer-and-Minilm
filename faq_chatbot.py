import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers.util import cos_sim

# Load the preprocessed data
df = pd.read_csv('cleaned_FAQ.csv')

# Load the fine-tuned model
model = SentenceTransformer('fine_tuned_model')

# Encode the questions
question_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)

def find_most_similar_question(user_question, question_embeddings, df, model):
    try:
        # Encode the user's question
        user_question_embedding = model.encode(user_question, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = cos_sim(user_question_embedding, question_embeddings)[0]
        
        # Find the index of the most similar question
        most_similar_idx = torch.argmax(similarities).item()
        
        return df.iloc[most_similar_idx]['Answer'], similarities[most_similar_idx].item()
    except Exception as e:
        return str(e), 0.0

def chatbot():
    print("Welcome to the HRM Chatbot. Let's Lead.")
    while True:
        user_input = input("Query: ")
        if user_input.lower() == 'exit':
            print("Hope you don't face any query further")
            break
        
        # Find the most similar question and answer
        answer, similarity = find_most_similar_question(user_input, question_embeddings, df, model)
        
        if similarity < 0.5:  # Threshold for similarity
            print("Solution: I'm not sure about the answer to that. Can you please rephrase?")
        else:
            print(f"Solution: {answer}")

if __name__ == "__main__":
    chatbot()
