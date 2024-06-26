from langchain_openai import ChatOpenAI
from textwrap import dedent
from .base import BaseAgent
import os

from dotenv import load_dotenv
load_dotenv(override=True)

llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")

class HostAgent(BaseAgent):
    def __init__(self, client, GAME_TOPIC):
        super().__init__(client)
        self.topic = GAME_TOPIC  
        self.question_count = 0 
        self.role_description = dedent(
            f"""You are a virtual assistant programmed to serve as the host in a {os.getenv("N_QUESTIONS")}-question, yes-or-no guessing game.

            Game Structure:
            - The game automatically selects a secret object or living thing, referred to as the "topic."
            - You, the "host," know the topic but do not disclose it to the "guesser" until the game concludes.
            - The "guesser" tries to identify the topic by asking yes-or-no questions, to which you must respond directly with "Yes" or "No."
            - The guesser has up to {os.getenv("N_QUESTIONS")} questions and guesses to determine the topic correctly and win the game.

            Role:
            As the host, your role is to provide clear, straightforward yes-or-no answers to the guesser's inquiries about the topic, which for this
            instance is set as: {GAME_TOPIC}. It is crucial that you do not offer any additional information or hints beyond a simple "Yes" or "No" response.

            Examples:
            - Guesser: Is the topic an animal?
            - Assistant: Yes

            - Guesser: Is the topic a plant?
            - Assistant: No

            - Guesser: Is the topic a household object?
            - Assistant: Yes

            You must adhere strictly to yes-or-no answers to maintain the integrity of the game.

            """
        )
        assert GAME_TOPIC in self.role_description

        self.responses = [{"role": "system", "content": self.role_description}]

    def get_response(self, guesser_message):
        """
        This method takes a guesser's message and manages the chat history.
        It then makes a chat completion request to generate a response based on the current game state.
        """
        self.responses.append({"role": "user", "content": guesser_message})
        
        if self.question_count >= int(os.getenv("N_QUESTIONS")):
            self.responses.append({"role": "system", "content": "Game Over"})
            return "Game Over. You've reached the maximum number of questions."
        

        # Process the message with historical chat context
        response = self.chat_completion_request(self.responses)

        # If a response is successfully generated, append it to the chat history
        if response:
            self.responses.append({"role": "assistant", "content": response.choices[0].message.content})
            self.question_count += 1
            return response.choices[0].message.content
        else:
            return "An error occurred. Please try asking your question again."

    
