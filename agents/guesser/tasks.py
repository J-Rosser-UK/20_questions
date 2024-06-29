import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatCohere
from langchain.agents import load_tools
from textwrap import dedent
from pprint import pprint
from dotenv import load_dotenv
load_dotenv(override=True)


class GuesserTasks:

    def __init__(self, senior_question_writer_and_planner, senior_analyst, manager):
        
        self.current_guess = None
        
        self._initialize_plan_and_ask_a_question_task(senior_question_writer_and_planner)

        self._initialize_make_a_guess_task(senior_analyst, self.plan_and_ask_a_question_task)

        


    def _initialize_plan_and_ask_a_question_task(self, senior_question_writer_and_planner):

        self.plan_and_ask_a_question_description = dedent("""
            As a team of guessers in a game of {n_questions} questions, your role is to determine the secret topic
            chosen by the host. The topic can be any object or living thing. You will ask a single
            yes-or-no question each round to narrow down the possibilities and make informed guesses.
            Remember, you have up to {n_questions} questions and guesses combined to identify the topic but you
            must only ask ONE yes-or-no question each round. Be strategic in your questioning and guessing.
            Use strategic questions to efficiently narrow down the potential options.

            Here is the history of the conversation so far:
            {conversation_history}
        """)
          

        self.plan_and_ask_a_question_task = Task(
            description=(self.plan_and_ask_a_question_description.format(
                n_questions=os.getenv("N_QUESTIONS"),
                conversation_history=[]
            )),
            expected_output='A single yes-or-no question, aimed at discovering the secret topic.',
            agent=senior_question_writer_and_planner,
            tools=[],
            human_input=False
        )


    def _initialize_make_a_guess_task(self, senior_analyst, plan_and_ask_a_question_task):

        self.make_a_guess_description = dedent("""
            Based on the questions asked and the responses received so far, you need to make an informed guess 
            about the secret topic chosen by the host. Consider all the information you have gathered 
            through the yes-or-no questions and use logical reasoning to deduce the most likely answer.

            You should review the questions you have asked and the answers you have received, then use this 
            information to make a single-word guess about the secret topic. Remember, the guess should be 
            a specific object or living thing, such as "Elephant" or "Table".

            Here is the plan and question you asked in the previous task:
            {plan_and_ask_a_question_task}
        """)
     
        self.make_a_guess_description = self.make_a_guess_description.format(
            plan_and_ask_a_question_task=self.plan_and_ask_a_question_task
        )

        self.make_a_guess_task = Task(
            description=(self.make_a_guess_description.format(
                plan_and_ask_a_question_task=self.plan_and_ask_a_question_task
            )),
            expected_output='A single word representing the guess. E.g. "Elephant" or "Table".',
            agent=senior_analyst,
            tools=[],
            context=[self.plan_and_ask_a_question_task],
            callback=self.make_a_guess_callback,
            human_input=False
        )

    def plan_and_ask_a_question(self, conversation_history):
        """ Update the conversation history in the description of the plan_and_ask_a_question task and return the task."""

        self.plan_and_ask_a_question_task.description = self.plan_and_ask_a_question_description.format(
            n_questions=os.getenv("N_QUESTIONS"),
            conversation_history=conversation_history
        )

        return self.plan_and_ask_a_question_task
    
    def make_a_guess(self, plan_and_ask_a_question_task):
        """ Update the description of the make_a_guess task with the plan_and_ask_a_question task and return the task."""

        self.make_a_guess_task.description = self.make_a_guess_description.format(
            plan_and_ask_a_question_task=plan_and_ask_a_question_task
        )
        return self.make_a_guess_task
    
    def make_a_guess_callback(self, output):
        self.current_guess = output.exported_output
        print("The guesser has made a guess. The guesser's guess is: ", self.current_guess)

    def get_tasks(self):
        return [self.make_a_guess_task, self.plan_and_ask_a_question_task]

    
