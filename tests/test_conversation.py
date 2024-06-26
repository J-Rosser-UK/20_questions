import sys
sys.path.append("")

import unittest
from unittest.mock import Mock, patch
from conversation import Conversation
from crewai import Agent, Task, Crew, Process
from agents import HostAgent, PlanningAgent
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI
import os


class TestConversation(unittest.TestCase):
    def setUp(self):
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.topic = "Pencil"
        
        self.mock_host = HostAgent(client = client, GAME_TOPIC=self.topic)
        self.mock_host.history = []
        self.mock_host.add_message = Mock()

        self.mock_guesser = PlanningAgent()

      

        # Create an instance of Conversation
        self.conversation = Conversation(host=self.mock_host, guesser=self.mock_guesser)

    def test_initialization(self):
        # Check if the host and guesser are correctly assigned
        self.assertEqual(self.conversation.host, self.mock_host)
        self.assertEqual(self.conversation.guesser_agent, self.mock_guesser)
        # Check if the guesser crew is properly set up
        self.assertIsInstance(self.conversation.guesser_crew, Crew)

    def test_run_conversation(self):
     
        # Generate the conversation generator
        conversation_gen = self.conversation.run_conversation()
        
        # Test the first message yield
        first_message = next(conversation_gen)
        self.assertIn(f"Starting a conversation on the topic: {self.topic}", first_message[0])

   

if __name__ == '__main__':
    unittest.main()
