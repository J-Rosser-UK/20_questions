from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
import random
import os
from abc import ABC, abstractmethod



class BaseAgent(ABC):
    def __init__(self, client: OpenAI, responses:list = []):
        self.responses = responses
        self.client = client
        self.history:list = []

    def add_message(self, sender:str, message:str):
        self.history.append({"role": sender, "content": message})

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages:list, model:str="gpt-3.5-turbo"):
        try:
            response = self.client.chat.completions.create(model=model,
            messages=messages)
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return None
    
    @abstractmethod
    def get_response(self, conversation_history):
        response = self.chat_completion_request(conversation_history)
        if response and "choices" in response:
            return response.choices[0].message.content
        return random.choice(self.responses)

    