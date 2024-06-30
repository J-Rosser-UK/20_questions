import gradio as gr
from conversation import Conversation
from openai import OpenAI
import os
import mlflow
from dotenv import load_dotenv
load_dotenv(override=True)
from eval.evaluate import Evaluate

eval = Evaluate()

def start_conversation(topic:str, api_key:str = None):
    # Set the tracking URI to the `mlruns` directory
    mlflow.set_tracking_uri("file:./mlruns")

    # Set the experiment name
    experiment_name = "Gradio Conversation"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        # Initialize the conversation
        convo = []

        # Initialize the OpenAI client with the given API key or from environment variables
        client = OpenAI(api_key=api_key) if (api_key and len(api_key) > 0) else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize the topic of the conversation
        if not topic or len(topic) == 0:
            topic = "Pencil"
            convo.append((f"No topic provided, setting default topic to '{topic}'.", None))
            yield convo

        # Initialize the conversation
        conversation = Conversation(client, topic)
        conversation.mlflow_log()

        # Start the conversation
        for (host_msg, guesser_msg) in conversation.run_conversation():

            if host_msg:
                print("host_msg: ", host_msg)
            if guesser_msg:
                print("guesser_msg: ", guesser_msg)
            
            convo.append((host_msg, guesser_msg))
            yield convo
        
        try:
            eval.mlflow_log(conversation.host.topic, conversation.guesser_crew.guesses, conversation.guesser_crew.questions)
        except Exception as e:
            print(f"A mlflow error occurred: {e}")
            yield f"A mlflow error occurred: {e}", None


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