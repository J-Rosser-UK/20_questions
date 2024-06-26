import sys
sys.path.append("")

import unittest
from agents.guesser import GuesserAgent, PlanningAgent, PlanningTask, GuesserTask
from crewai import Crew, Process, Task
from textwrap import dedent
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class TestGuesserAgent(unittest.TestCase):
    def setUp(self):

        self.guesser = PlanningAgent()

        guesser_task = GuesserTask(
            guesser=self.guesser,
            history=[]
        )

        
        self.guesser_crew = Crew(
            agents=[self.guesser],
            tasks=[guesser_task],
            process=Process.sequential,  
            memory=True,
            cache=True,
            max_rpm=100,
            manager_agent=self.guesser
        )

    def test_agent_can_remember_game_instructions(self):
        """ Test if the agent can remember the game instructions."""
        
        # Create a new task for this test
        remember_task = Task(
                description=(dedent(f"""
                    Your task is to as clearly as possible describe the game instructions in 1 sentence.
                """)),
            expected_output='A single sentence describing the game instructions.',
            tools=[],
            agent=self.guesser,
            human_input=False
        )

        # Give the crew a new task which is to remember the game instructions
        self.guesser_crew.tasks = [remember_task]

        # Retrieve the response from the crew
        guesser_response = self.guesser_crew.kickoff()

        # Check if the game topic is correctly remembered and incorporated in the role description
        self.assertIn(str(os.getenv("N_QUESTIONS")), guesser_response)
        self.assertIn("question", guesser_response.lower())
        self.assertIn("game", guesser_response.lower())

    


if __name__ == "__main__":
    unittest.main()
