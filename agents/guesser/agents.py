import random
import os
import mlflow
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import load_tools
import json
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv(override=True)

llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")

class Agents:

    def __init__(self):

        self.senior_analyst_config = {
            "role": "Senior Analyst",
            "goal": "To analyze the responses to each yes-or-no question, using logical reasoning and data analysis to narrow down the possibilities and make accurate guesses about the secret topic.",
            "backstory": dedent("""
                With a Ph.D. in Data Science, you have a background in analyzing complex datasets and deriving insights from minimal information.
                Formerly a lead data analyst at a top research institute, you are renowned for your meticulous approach to problem-solving and your
                ability to see patterns where others see chaos. This knack for analytical thinking makes you an invaluable asset in guessing the secret
                topic, either an object or living thing, efficiently.
            """),
            "allow_delegation": False,
        }

   

        self.senior_question_writer_and_planner_config = {
            "role": "Senior Question Writer and Planner",
            "goal": "To craft strategic yes-or-no questions that will most effectively narrow down the range of possible topics, ensuring that each question asked maximizes the information gained.",
            "backstory": dedent("""
                A seasoned journalist turned strategist, you have spent years mastering the art of asking the right questions.
                With a career that began in investigative journalism, you have a keen eye for details and a talent for planning.
                Your expertise lies in formulating questions that uncover hidden truths, making you perfect for devising the right
                inquiries to zero in on the secret topic.                 
            """),
            "allow_delegation": False,
        }

       

        self.manager_config = {
            "role": "Manager",
            "goal": "To manage the crew and ensure the tasks are completed efficiently, coordinating the efforts of the Researcher and Writer to achieve the best possible outcome in the game.",
            "backstory": dedent("""
                An experienced project manager with a background in both the tech industry and team leadership, you have overseen numerous high-stakes
                projects to successful conclusions. Known for your ability to keep teams focused and organized, you excel at breaking down complex tasks
                into manageable steps and ensuring smooth collaboration. Your role is to guide the Researcher and Writer, making sure each task is executed
                flawlessly and on time.             
            """),
            "allow_delegation": False
        }


    def senior_analyst(self):

        agent = Agent(**self.senior_analyst_config)

        return agent
    
    def senior_question_writer_and_planner(self):

        return Agent(**self.senior_question_writer_and_planner_config)

    def manager(self):
        
        return Agent(**self.manager_config)
    

    def mlflow_log_agent(self, agent_config:dict):
        """ Retrieves agent role for file name and converts dict to json for logging."""
        role = agent_config["role"].replace(" ", "_").lower()

        agent_config = json.dumps(agent_config)

        mlflow.log_text(agent_config, artifact_file=f"guesser_agent_{role}_config.json")

