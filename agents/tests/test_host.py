import sys
sys.path.append("")

import unittest
from agents.host import HostAgent
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class TestHostAgent(unittest.TestCase):
    def setUp(self):
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.topic = "Elephant"
        self.agent = HostAgent(self.client, topic=self.topic)

    def test_agent_remembers_topic(self):
        """ Test if the agent can remember the game topic."""
        
        response = self.agent.get_response("The game has ended. What was the game topic? ")

        self.assertIn(self.topic, response)

    
if __name__ == "__main__":
    unittest.main()
