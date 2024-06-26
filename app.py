import gradio as gr
from conversation import Conversation
from agents import HostAgent, GuesserAgent, PlanningAgent
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)



def start_conversation(topic, api_key=None):
    if api_key:
        client = OpenAI(api_key)
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    convo = []
    if not topic or len(topic) == 0:
        topic = "Pencil"
        convo.append((f"No topic provided, setting default topic to '{topic}'.", None))
        
        yield convo

    host = HostAgent(client = client, GAME_TOPIC=topic)
    guesser = PlanningAgent()

    conversation = Conversation(host, guesser)

    
    for (host_msg, guesser_msg) in conversation.run_conversation():

        if host_msg:
            print("host_msg: ", host_msg)
        if guesser_msg:
            print("guesser_msg: ", guesser_msg)
        
        convo.append((host_msg, guesser_msg))
        yield convo


with gr.Blocks() as game:
    gr.Markdown("# 20 Question Guessing Game Between Two Agents")

    with gr.Row():
        topic_input = gr.Textbox(label="Enter the topic of conversation:")
        api_key_input = gr.Textbox(label="Enter OpenAI API Key (Optional):")
        start_button = gr.Button("Start Conversation")

    chatbot = gr.Chatbot()

    # Define how messages are displayed based on the sender's role
    start_button.click(fn=start_conversation, inputs=[topic_input, api_key_input], outputs=chatbot)

    reset_button = gr.Button("Reset Conversation")
    reset_button.click(fn=lambda: [], inputs=[], outputs=chatbot)

if __name__ == "__main__":

    game.launch()