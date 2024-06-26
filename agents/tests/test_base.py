import sys
sys.path.append("")

import unittest
from agents.host import BaseAgent
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import unittest
from openai import OpenAI

class TestBaseAgent(unittest.TestCase):
    class TestAgent(BaseAgent):
        def get_response(self, conversation_history):
            return super().get_response(conversation_history)

    def setUp(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        self.agent = self.TestAgent(client=self.client, responses=["Hi", "Hello", "Hey"])

    def test_initialization(self):
        self.assertEqual(self.agent.responses, ["Hi", "Hello", "Hey"])
        self.assertEqual(self.agent.client, self.client)
        self.assertEqual(len(self.agent.history), 0)

    def test_add_message(self):
        self.agent.add_message("user", "How are you?")
        self.agent.add_message("agent", "I am fine, thanks.")
        self.assertEqual(len(self.agent.history), 2)
        self.assertEqual(self.agent.history[0]['role'], "user")
        self.assertEqual(self.agent.history[0]['content'], "How are you?")
        self.assertEqual(self.agent.history[1]['role'], "agent")
        self.assertEqual(self.agent.history[1]['content'], "I am fine, thanks.")

    def test_chat_completion_request(self):
        messages = [
            {"role": "system", "content": "Start"},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am an AI trained by OpenAI."}
        ]
        response = self.agent.chat_completion_request(messages)
        self.assertIsNotNone(response)

    def test_get_response(self):
        self.agent.history = [
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am an AI trained by OpenAI."}
        ]
        response = self.agent.get_response(self.agent.history)
        self.assertIn(response, ["Hi", "Hello", "Hey"])

if __name__ == '__main__':
    unittest.main()

        
   
