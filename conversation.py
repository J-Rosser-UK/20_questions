import time
from crewai import Crew, Process
from openai import OpenAI
from agents.guesser import *
from agents.host import *
import os

class Conversation:

    def __init__(self, client:OpenAI, topic:str):

        self.host = HostAgent(client = client, GAME_TOPIC=topic)
        self.guesser_agent = PlanningAgent()
        
        
        guesser_task = GuesserTask(
            guesser=self.guesser_agent,
            history=self.host.history 
        )

        
        self.guesser_crew = Crew(
            agents=[self.guesser_agent],
            tasks=[guesser_task],
            process=Process.sequential,  
            memory=True,
            cache=True,
            max_rpm=100,
            manager_agent=self.guesser_agent
        )



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
            
        yield "Conversation ended.", None

        

    def round(self):
        # Host responds
        host_response = self.host.get_response(guesser_message=self.previous_msg)
        self.host.add_message("host", host_response)
        yield host_response, None

        # Update the guesser task with the current history
        self.guesser_crew.tasks = [GuesserTask(
            guesser=self.guesser_agent,
            history=self.host.history  # Pass the history to the task
        )]

        # Guesser asks a question
        guesser_response = self.call_crew_with_backoff()
        self.host.add_message("guesser", guesser_response)
        self.previous_msg = guesser_response
        yield None, guesser_response
           
    def call_crew_with_backoff(self):
        for i in range(1, 10):
            try:
                guesser_response = self.guesser_crew.kickoff()
                print("guesser_response: ", guesser_response)
                if not guesser_response or "limited to 10 API calls / minute" in guesser_response:
                    raise Exception("Crew call failed")
                return guesser_response
            except Exception as e:
                print(f"Attempt {i} failed with error: {e}")
                
            time.sleep(10*i)
        raise Exception("Crew call failed after 3 attempts")