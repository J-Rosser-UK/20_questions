import time
from crewai import Crew, Process
from openai import OpenAI
from agents.guesser import *
from agents.host import *
from agents.guesser.crew import GuesserCrew
import os
import string



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
            game_over = False
            
            round_number = 0
            host_save, guesser_save = None, None 
            while not game_over and round_number < int(os.getenv("N_QUESTIONS")):
                 
                for host_response, guesser_response in self.round():
                    
                    host_save = host_response if host_response else host_save
                    
                    if self._guessed_correctly(host_save, guesser_save, self.guesser_crew.guesses):
                        yield "The guesser has guessed the topic correctly!", None
                        game_over = True
                        break
                    
                    if not game_over:
                        guesser_save = guesser_response if guesser_response else guesser_save
                        yield host_response, guesser_response
                    else:
                        break
                    

                round_number += 1
        
                

        except Exception as e:
            print(f"An error occurred: {e}")
            yield f"An error occurred: {e}", None
            
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

    def _strip(self, word):
        # Convert the word to lowercase
        word = word.lower()
        # Create a translation table that maps each punctuation character to None
        translator = str.maketrans('', '', string.punctuation)
        # Use the translation table to remove punctuation
        stripped_word = word.translate(translator)
        return stripped_word
    

    def _guessed_correctly(self, host_save, guesser_save, previous_guesses) -> bool:
        print("GUESS ANALYSIS", host_save, guesser_save, previous_guesses)

        if len(previous_guesses) > 0 and guesser_save and host_save:
            # Check if the host has confirmed the guess
            if "yes" in host_save.lower() and len(host_save) > 4:
                # Check if the guess is the same as the topic
                print(self._strip(self.host.topic) in self._strip(previous_guesses[-1]))
                if self._strip(self.host.topic) in self._strip(previous_guesses[-1]):
                    # If the topic is in the last guess
                    if self._strip(self.host.topic) in guesser_save:
                        return True
            
        return False


    def mlflow_log(self):
        """
            The MLflow logging is decoupled from the core functionality, ensuring that the class can be fully utilized 
            without the need for MLflow. All MLflow logging functionalities are contained within separate methods and 
            are called explicitly when needed.
        """
        
        self.host.mlflow_log()
        self.guesser_crew.mlflow_log()

        

        
        

           
    