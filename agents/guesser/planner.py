
import random
import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain.agents import load_tools
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv(override=True)

llm=ChatCohere(temperature=0.3)

def PlanningAgent():
    return Agent(
        role=f'Planner for a {os.getenv("N_QUESTIONS")} Question Guessing Game',
        goal=f'To develop strategies and determine the next best question to ask in the game based on previous chat interactions',
        backstory=dedent(f"""
            This agent is designed to assist in planning and strategizing the game of {os.getenv("N_QUESTIONS")} questions.
            It analyzes the game's progress, the host's responses, and the guesser's questions to recommend the
            next strategic question to narrow down the possibilities effectively.
        """),
        verbose=True,
        allow_delegation=True,
        llm=llm,
    )

def PlanningTask(planner:Agent):
    # Planning Task
    return Task(
        description=(dedent(f"""
            Analyze the chat history so far in a game of {os.getenv("N_QUESTIONS")} questions and plan the next best step.
            Your task is to recommend the next question to be asked by the guesser that will most effectively
            narrow down the possibilities based on previous responses.
        """)),
        expected_output='A recommended question or strategy to follow next in the game.',
        tools=[],
        agent=planner,
        human_input=False
    )