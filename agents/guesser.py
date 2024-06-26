from .base import BaseAgent
import random
import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain.agents import load_tools
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv(override=True)


llm=ChatCohere(temperature=0.3)


def GuesserAgent():
    return Agent(
        role='Guesser of a 20 Question Guessing Game',
        goal='To correctly guess the topic of the game by asking yes or no questions to the host and making guesses each round.',
        backstory=dedent(f"""
            You will be playing the game of 20 questions.

            The game works as follows:

            * The game begins with an automatically selected object or living thing, which we’ll call the “topic” of the game
            * The first player, who we’ll call the “host”, is the only one who knows the topic, and does not reveal the topic to the other player until the end of the game

            * The other player – the “guesser” – needs to guess the topic
            * To learn more about what the host is thinking about, the guesser can ask yes-or-no questions. The host will then reply accordingly
            * As options narrow down, the guesser can make direct guesses. If the guesser correctly guesses the topic, they win!
            * The guesser has up to 20 total questions and guesses to win

            Role:

            Your role will be to be the guesser of the game, asking the host a single yes-or-no question each round to guess the secret topic.
        """),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def GuesserTask(guesser: Agent, history):
    return Task(
        description=(dedent(f"""
            As the guesser in a game of 20 questions, your role is to determine the secret topic
            chosen by the host. The topic can be any object or living thing. You will ask a single
            yes-or-no question each round to narrow down the possibilities and make informed guesses.
            Remember, you have up to 20 questions and guesses combined to identify the topic but you
            must only ask ONE yes-or-no question each round. Be strategic in your questioning and guessing.
            Use strategic questions to efficiently narrow down the potential options.

            Here is the history of the conversation so far:
            {history}
        """)),
        expected_output='A single yes-or-no question, aimed at discovering the secret topic.',
        tools=[],
        agent=guesser,
        human_input=False
    )



def PlanningAgent():
    return Agent(
        role='Planner for a 20 Question Guessing Game',
        goal='To develop strategies and determine the next best question to ask in the game based on previous chat interactions',
        backstory=dedent("""
            This agent is designed to assist in planning and strategizing the game of 20 questions.
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
        description=(dedent("""
            Analyze the chat history so far in a game of 20 questions and plan the next best step.
            Your task is to recommend the next question to be asked by the guesser that will most effectively
            narrow down the possibilities based on previous responses.
        """)),
        expected_output='A recommended question or strategy to follow next in the game.',
        tools=[],
        agent=planner,
        human_input=False
    )