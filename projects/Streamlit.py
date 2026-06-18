import streamlit as st
from transformers import pipeline

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="TinyLlama Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 TinyLlama AI Chatbot")

# ----------------------------------
# Load Model Once
# ----------------------------------

@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    return generator

generator = load_model()

# ----------------------------------
# Session State
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------
# Display Previous Messages
# ----------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ----------------------------------
# User Input
# ----------------------------------

user_input = st.chat_input("Ask me anything...")

if user_input:

    # Show User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Build Prompt
    prompt = f"""
You are a helpful AI assistant.

User: {user_input}

Assistant:
"""

    # Generate Response
    response = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    generated_text = response[0]["generated_text"]

    # Extract Assistant Response
    answer = generated_text.split("Assistant:")[-1].strip()

    # Save Assistant Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Display Assistant Message
    with st.chat_message("assistant"):
        st.write(answer)