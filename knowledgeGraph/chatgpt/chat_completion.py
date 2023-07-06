import os
import openai
import streamlit as st

# Load your API key from an environment variable or secret management service
openai.api_key = st.secrets["chatgpt_api_key"]

message = """
최신 영화를 추천해줍니다.
- 슈퍼마리오 브라더스
- 존윅
- 노잼 

위의 영화를 먼저 추천해줘
"""

messages=[{"role": "system", "content": message }]


def ask(q):
    q = {"role" : "user" , "content" : q}
    messages.append(q)

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = messages
    )

    res = response.to_dict_recursive()
    bot_text  = response['choices'][0]['message']['content']
    bot_input = {"role": "assistant", "content": bot_text }

    messages.append(bot_input)

    return bot_text

while True:
    user_input = user_input("프롬프트 입력: ")
    bot_resp = ask(user_input)
    print("-"*30)
    print(f"프롬프트: {bot_resp}")
    print(f"응답결과: {bot_resp}")
