from langchain_openai import ChatOpenAI
from textwrap import dedent
from .base import BaseAgent

from dotenv import load_dotenv
load_dotenv(override=True)

llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")

class HostAgent(BaseAgent):
    def __init__(self, client, GAME_TOPIC):
        super().__init__(client)
        self.topic = GAME_TOPIC  # The secret topic for the 20 questions game
        self.question_count = 0  # Track the number of questions asked
        self.role_description = dedent(
            f"""You are a helpful assistant acting as the host of a 20 question, yes-or-no guessing game.

            The game works as follows:

            * The game begins with an automatically selected object or living thing, which well call the “topic” of the game
            * The first player, who wll call the “host”, is the only one who knows the topic, and does not reveal the topic to the other player until the end of the game

            * The other player the “guesser” needs to guess the topic
            * To learn more about what the host is thinking about, the guesser can ask yes-or-no questions. The host will then reply accordingly
            * As options narrow down, the guesser can make direct guesses. If the guesser correctly guesses the topic, they win!
            * The guesser has up to 20 total questions and guesses to win

            Role:

            Your role will be to be host of the game, where the topic is: {GAME_TOPIC}

            As the host of the game, your primary role is to respond accurately to the guessers yes-or-no questions.

            You will never reveal the topic to the guesser, but you will provide truthful answers to their questions.

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
        
        if self.question_count >= 20:
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

    
