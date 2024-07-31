import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", message="Unsupported Windows version (2016server). ONNX Runtime supports Windows 10 and above, only.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Function to load documents and embed them
def load_and_embed_documents():
    documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    print(f"Loading {file}")
                    loader = PDFMinerLoader(os.path.join(root, file))
                    doc = loader.load()
                    documents.extend(doc)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    if not documents:
        print("No documents were loaded. Please check your PDF files.")
        return None, None

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embedded_texts = embeddings.embed_documents([text.page_content for text in texts])
        return texts, embedded_texts
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        return None, None

# Custom retriever function
def my_retriever(texts):
    def retrieve(query):
        relevant_documents = [text.page_content for text in texts if query.lower() in text.page_content.lower()]
        return relevant_documents
    return retrieve

# Function to set up the question answering pipeline
def setup_qa_pipeline():
    texts, embedded_texts = load_and_embed_documents()
    
    # Set up the language model pipeline
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Create retriever function
    retriever = my_retriever(texts)
    
    # Set up the QA pipeline with the language model and custom retriever
    qa = RetrievalQA.from_pipeline(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# Function to process user questions and generate answers
def process_question(qa, question):
    try:
        generated_text = qa({'query': question})
        answer = generated_text['result']
        return answer
    except Exception as e:
        print(f"Error processing question: {e}")
        return "Sorry, I couldn't find an answer to your question."

# Main function to run the chatbot
def main():
    qa = setup_qa_pipeline()

    print("Welcome to PDF Chatbot!")
    print("Ask me any question about the documents in the 'docs' folder. Type 'exit' to end.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Exiting PDF Chatbot. Goodbye!")
            break

        answer = process_question(qa, user_input)
        print("PDF Chatbot:", answer)

if __name__ == "__main__":
    main()
