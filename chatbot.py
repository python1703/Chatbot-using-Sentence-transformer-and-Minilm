from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=512,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        top_p=0.9,
        top_k=50,
        temperature=0.7,
        do_sample=True  # Enable sampling
    )
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model_save_path = "fine-tuned-model"
    tokenizer_save_path = "fine-tuned-tokenizer"
    
    model = AutoModelForCausalLM.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit']:
            break
        response = generate_response(prompt, model, tokenizer)
        while len(response.split()) < 20:  # Arbitrary threshold for brevity
            print(f"Bot: {response}")
            follow_up = input("Can you provide more details or clarify further? ")
            response = generate_response(follow_up + " " + response, model, tokenizer)
        print(f"Bot: {response}")
