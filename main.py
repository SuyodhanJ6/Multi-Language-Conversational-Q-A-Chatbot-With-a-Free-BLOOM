# main.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Function to initialize BLOOM model
def initialize_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Function to generate response
def generate_response(model, tokenizer, user_input, max_length=50):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    st.title("BLOOM Chatbot")

    # Choose BLOOM model
    model_name = st.selectbox("Select BLOOM Model", ["bigscience/bloom-560m", "other/bloom-model"])
    tokenizer, model = initialize_model(model_name)

    st.sidebar.markdown("## Conversation")

    # Interactive chat
    user_input = st.text_input("You:")
    if user_input:
        response = generate_response(model, tokenizer, user_input)
        st.text_area("Chatbot:", value=response, height=100)

if __name__ == "__main__":
    main()