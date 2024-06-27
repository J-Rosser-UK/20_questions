
import random
import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain.agents import load_tools
from textwrap import dedent
from .tasks import GuesserTasks
from .agents import Agents
from dotenv import load_dotenv
load_dotenv(override=True)


llm=ChatCohere(temperature=0.3)

class GuesserCrew:

    def __init__(self):
        
        self.agents = Agents()
        self.tasks = GuesserTasks()

    def run(self, conversation_history):
    
        senior_question_writer_and_planner, senior_analyst, manager = self.initialize_agents()

        
        plan_and_ask_a_question = self.tasks.plan_and_ask_a_question(
            agent = senior_question_writer_and_planner,
            conversation_history = conversation_history
        )

        make_a_guess = self.tasks.make_a_guess(
            agent = senior_question_writer_and_planner,
            plan_and_ask_a_question_task = plan_and_ask_a_question
        )
        
        crew = Crew(
            agents=[senior_question_writer_and_planner, senior_analyst, manager],
            tasks=[plan_and_ask_a_question],
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=100,
            
        )

        result = crew.kickoff()

        crew = Crew(
            agents=[senior_question_writer_and_planner, senior_analyst, manager],
            tasks=[make_a_guess],
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=100,
            
        )

        guess = crew.kickoff()

        print("Guess: ", guess)

        return result

    def initialize_agents(self):
        senior_question_writer_and_planner = self.agents.senior_question_writer_and_planner()
        senior_analyst = self.agents.senior_analyst()
        manager = self.agents.manager()

        return senior_question_writer_and_planner, senior_analyst, manager
        