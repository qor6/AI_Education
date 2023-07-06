import os
import openai
import streamlit as st
#ctrl + F5 누르기

# Load your API key from an environment variable or secret management service
openai.api_key ="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"
#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

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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = messages,
        max_tokens=100,  # 생성된 응답의 최대 토큰 수
        n=1,  # 생성할 응답의 수
        stop=None,  # 생성된 응답을 중단할 문자열
        temperature=0.7,  # 생성의 다양성을 조절하는 온도 값
        top_p=1.0,  # 다양성을 조절하는 top-p 샘플링의 임계값
        frequency_penalty=0.0,  # 빈도 페널티 파라미터
        presence_penalty=0.0  # 존재 페널티 파라미터
    )

    res = response.to_dict_recursive()
    bot_text  = response['choices'][0]['message']['content']
    bot_input = {"role": "assistant", "content": bot_text }

    messages.append(bot_input)

    return bot_text

while True:
    user_input = input("Prompt input: ")
    bot_resp = ask(user_input)
    print("-"*30)
    print(f"Prompt: {bot_resp}")
    print(f"respon result: {bot_resp}")
