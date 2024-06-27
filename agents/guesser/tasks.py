import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain.agents import load_tools
from textwrap import dedent

from dotenv import load_dotenv
load_dotenv(override=True)


class GuesserTasks:
    
    def plan_and_ask_a_question(self, agent:Agent, conversation_history):
        return Task(
            description=(dedent(f"""
                As a team of guessers in a game of {os.getenv("N_QUESTIONS")} questions, your role is to determine the secret topic
                chosen by the host. The topic can be any object or living thing. You will ask a single
                yes-or-no question each round to narrow down the possibilities and make informed guesses.
                Remember, you have up to {os.getenv("N_QUESTIONS")} questions and guesses combined to identify the topic but you
                must only ask ONE yes-or-no question each round. Be strategic in your questioning and guessing.
                Use strategic questions to efficiently narrow down the potential options.

                Here is the history of the conversation so far:
                {conversation_history}
            """)),
            expected_output='A single yes-or-no question, aimed at discovering the secret topic.',
            tools=[],
            agent=agent,
            human_input=False
        )
    
    def make_a_guess(self, agent:Agent, plan_and_ask_a_question_task):
        return Task(
            description=dedent(f"""
                Based on the questions asked and the responses received so far, you need to make an informed guess 
                about the secret topic chosen by the host. Consider all the information you have gathered 
                through the yes-or-no questions and use logical reasoning to deduce the most likely answer.

                You should review the questions you have asked and the answers you have received, then use this 
                information to make a single-word guess about the secret topic. Remember, the guess should be 
                a specific object or living thing, such as "Elephant" or "Table".

                Here is the plan and question you asked in the previous task:
                {plan_and_ask_a_question_task}
            """),
            expected_output='A single word representing the guess. E.g. "Elephant" or "Table".',
            agent=agent,
            tools=[],
            context=[plan_and_ask_a_question_task],
            callback=self.make_a_guess_callback,
            human_input=False
        )
    
    def make_a_guess_callback(self, output):
        
        print("The guesser has made a guess. The guesser's guess is: ", output)