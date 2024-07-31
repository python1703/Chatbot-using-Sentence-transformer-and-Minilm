import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the fine-tuned model and tokenizer
model_name = 'fine_tuned_gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the preprocessed data
df = pd.read_csv('cleaned_FAQ.csv')

# Prepare TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Answer'])

def retrieve_relevant_answer(query, df, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    best_match_index = similarities.argmax()
    return df.iloc[best_match_index]['Answer']

def generate_answer(question, model, tokenizer, max_length=100):
    retrieved_answer = retrieve_relevant_answer(question, df, vectorizer, tfidf_matrix)
    input_text = f"Question: {question}\nAnswer: {retrieved_answer}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate the answer
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

    # Decode the output to get the answer
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text.split("Answer:")[1].strip()

def chatbot():
    print("Welcome to the HRM Chatbot. Let's Lead.")
    while True:
        user_input = input("Query: ")
        if user_input.lower() == 'exit':
            print("Hope you don't face any query further")
            break
        
        # Generate the answer using the fine-tuned model
        answer = generate_answer(user_input, model, tokenizer)
        print(f"Solution: {answer}")

if __name__ == "__main__":
    chatbot()
