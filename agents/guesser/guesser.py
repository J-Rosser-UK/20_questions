
import random
import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import load_tools
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv(override=True)


llm=ChatOpenAI(temperature=0.3)


def GuesserAgent():
    return Agent(
        role=f'Guesser of a {os.getenv("N_QUESTIONS")} Question Guessing Game',
        goal=f'To correctly guess the topic of the game by asking yes or no questions to the host and making guesses each round.',
        backstory=dedent(f"""
            You will be playing the game of {os.getenv("N_QUESTIONS")} questions.

            The game works as follows:

            * The game begins with an automatically selected object or living thing, which we’ll call the “topic” of the game
            * The first player, who we’ll call the “host”, is the only one who knows the topic, and does not reveal the topic to the other player until the end of the game

            * The other player – the “guesser” – needs to guess the topic
            * To learn more about what the host is thinking about, the guesser can ask yes-or-no questions. The host will then reply accordingly
            * As options narrow down, the guesser can make direct guesses. If the guesser correctly guesses the topic, they win!
            * The guesser has up to {os.getenv("N_QUESTIONS")} total questions and guesses to win

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
            As the guesser in a game of {os.getenv("N_QUESTIONS")} questions, your role is to determine the secret topic
            chosen by the host. The topic can be any object or living thing. You will ask a single
            yes-or-no question each round to narrow down the possibilities and make informed guesses.
            Remember, you have up to {os.getenv("N_QUESTIONS")} questions and guesses combined to identify the topic but you
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

