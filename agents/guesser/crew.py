
import random
import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain.agents import load_tools
from textwrap import dedent
from .tasks import GuesserTasks
import json
from .agents import Agents
import mlflow
from dotenv import load_dotenv
load_dotenv(override=True)


class GuesserCrew:

    def __init__(self):
        
        self.agents = Agents()
        self.senior_question_writer_and_planner, self.senior_analyst, self.manager = self.initialize_agents()
        self.tasks = GuesserTasks(self.senior_question_writer_and_planner, self.senior_analyst, self.manager)

        plan_and_ask_a_question = self.tasks.plan_and_ask_a_question(
            conversation_history = []
        )

        self.crew = Crew(
            agents=[self.senior_question_writer_and_planner, self.senior_analyst, self.manager],
            tasks=[plan_and_ask_a_question],
            process=Process.sequential,
            memory=True,
            cache=True,
            verbose=False,
            max_rpm=100,
            
        )
        

    def run(self, conversation_history):
        """ Run the Guesser Crew."""
        
        plan_and_ask_a_question = self.tasks.plan_and_ask_a_question(
            conversation_history = conversation_history
        )

        make_a_guess = self.tasks.make_a_guess(
            plan_and_ask_a_question_task = plan_and_ask_a_question
        )

        self.crew.tasks = [make_a_guess, plan_and_ask_a_question]
        
        result = self.crew.kickoff()

        return result

    def initialize_agents(self):
        """ Initialize the agents for the Guesser Crew."""

        senior_question_writer_and_planner = self.agents.senior_question_writer_and_planner()
        senior_analyst = self.agents.senior_analyst()
        manager = self.agents.manager()

        return senior_question_writer_and_planner, senior_analyst, manager
    
    def _mlflow_custom_serializer(self, obj):
        """ Custom serializer to handle serialization of Crew and Agent objects."""

        if isinstance(obj, dict):
            return self._mlflow_custom_serializer(obj)
        
        elif isinstance(obj, list):

            serialized_list = []
            for i in obj:
                serialized_list.append(self._mlflow_custom_serializer(i))

            return serialized_list
        
        elif isinstance(obj, Agent):
            return dict(obj)
        
        return str(obj)
    

    def mlflow_log(self):
        """
            The MLflow logging is decoupled from the core functionality, ensuring that the class can be fully utilized 
            without the need for MLflow. All MLflow logging functionalities are contained within separate methods and 
            are called explicitly when needed.
        """

        for task in self.tasks.get_tasks():

            role = task.agent.role.replace(" ", "_").lower()

            mlflow.log_text(json.dumps(dict(task), default=self._mlflow_custom_serializer), artifact_file=f"guesser_agent_{role}_tasks_config.json")

        mlflow.log_text(json.dumps(dict(self.crew), default=self._mlflow_custom_serializer), artifact_file=f"guesser_agent_crew_config.json")

        for agent in self.crew.agents:

            role = agent.role.replace(" ", "_").lower()

            mlflow.log_text(json.dumps(dict(agent), default=self._mlflow_custom_serializer), artifact_file=f"guesser_agent_{role}_config.json")
        