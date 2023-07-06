import os
import openai
import streamlit as st
from openai.error import OpenAIError
#https://platform.openai.com/docs/guides/images/usage
def clear_submit():
    st.session_state["submit"] = False
    
def draw(q):
    response = openai.Image.create(
    prompt=q,
    n=1,
    size="1024x1024"
    )
    return response['data'][0]['url']

# Load your API key from an environment variable or secret management service
openai.api_key = st.secrets["chatgpt_api_key"]

st.header("AI IMAGE")

query = st.text_area('AI 이미지 생성을 위한 텍스트를 입력하세요', value="Create a funny AI-generated image where a monkey is wearing a tutu and playing the guitar.", on_change=clear_submit)
button = st.button("submit")

if button or st.session_state.get("submit"):
    st.session_state["submit"] = True

    try:
        with st.spinner("Calling DALL-E API..."):
            image_url = draw(query)

        st.markdown("#### AI IMAGE")
        st.image(image_url)

    except OpenAIError as e:
        st.error(e._message)

