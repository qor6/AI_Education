import os
import openai

#key : #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"

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("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")

#링크 : https://platform.openai.com/docs/libraries
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])
print(response)
