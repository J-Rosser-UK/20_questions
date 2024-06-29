import time
from crewai import Crew, Process
from openai import OpenAI
from agents.guesser import *
from agents.host import *
from agents.guesser.crew import GuesserCrew
import os

class Conversation:

    def __init__(self, client:OpenAI, topic:str):

        llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")
        self.host = HostAgent(client = client, topic=topic)
        self.guesser_crew = GuesserCrew()
        


    def run_conversation(self):

        try:
            self.host.add_message("host", f"Starting a conversation on the topic:")

            yield self.host.history[-1]["content"] + f" {self.host.topic}", None  

            self.previous_msg = "Start the game."
            for round in range(0,int(os.getenv("N_QUESTIONS"))):  
                for host_response, guesser_response in self.round():
                    
                    yield host_response, guesser_response

        except Exception as e:
            print(f"An error occurred: {e}")
            yield "An error occurred: ", None
            
        yield f"Conversation ended. \nGuesses list: {self.guesser_crew.guesses} \nQuestions list: {self.guesser_crew.questions}", None

        

    def round(self):
        # Host responds
        host_response = self.host.get_response(guesser_message=self.previous_msg)
        self.host.add_message("host", host_response)
        yield host_response, None


        # Guesser asks a question
        guesser_response = self.guesser_crew.run(self.host.history)
        self.host.add_message("guesser", guesser_response)
        self.previous_msg = guesser_response
        yield None, guesser_response


    def mlflow_log(self):
        """
            The MLflow logging is decoupled from the core functionality, ensuring that the class can be fully utilized 
            without the need for MLflow. All MLflow logging functionalities are contained within separate methods and 
            are called explicitly when needed.
        """
        self.host.mlflow_log()
        self.guesser_crew.mlflow_log()

           
    