import sys
sys.path.append("")

import unittest
from agents.guesser import GuesserCrew
from crewai import Crew, Process, Task
from textwrap import dedent
import inflect
inflect_engine = inflect.engine()

import os
from dotenv import load_dotenv
load_dotenv(override=True)

class TestGuesserAgent(unittest.TestCase):
    def setUp(self):

        self.guesser = GuesserCrew()

        self.senior_question_writer_and_planner, self.senior_analyst, self.manager = self.guesser.initialize_agents()

        


    def test_agent_can_remember_game_instructions(self):
        """ Test if the agent can remember the game instructions."""
        
        # Create a new task for this test
        remember_task = Task(
                description=(dedent(f"""
                                    
                    As a team of guessers in a game of {os.getenv("N_QUESTIONS")} questions, your role is to determine the secret topic
                    chosen by the host. The topic can be any object or living thing. You will ask a single
                    yes-or-no question each round to narrow down the possibilities and make informed guesses.
                    Remember, you have up to {os.getenv("N_QUESTIONS")} questions and guesses combined to identify the topic but you
                    must only ask ONE yes-or-no question each round. Be strategic in your questioning and guessing.
                    Use strategic questions to efficiently narrow down the potential options.
                    
                    Your task is to as clearly as possible describe the game instructions in 1 sentence.
                """)),
            expected_output='A single sentence describing the game instructions.',
            tools=[],
            agent=self.manager,
            human_input=False
        )

        self.guesser_crew = Crew(
            agents=[self.senior_question_writer_and_planner, self.senior_analyst, self.manager],
            tasks=[remember_task],
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=100,
            
        )

        # Retrieve the response from the crew
        guesser_response = self.guesser_crew.kickoff()

        print(guesser_response)
        # Check if the game topic is correctly remembered and incorporated in the role description
        self.assertTrue(str(os.getenv("N_QUESTIONS")) in guesser_response.lower() or inflect_engine.number_to_words(os.getenv("N_QUESTIONS")) in guesser_response.lower())
        self.assertIn("question", guesser_response.lower())
        self.assertIn("game", guesser_response.lower())

    


if __name__ == "__main__":
    unittest.main()
