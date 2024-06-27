import mlflow
import gradio as gr
from conversation import Conversation
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)
import concurrent.futures
from tqdm import tqdm
from textwrap import dedent

# Set the tracking URI to the `mlruns` directory
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name
experiment_name = "Healthcheck"
mlflow.set_experiment(experiment_name)

def topic():
    """Generator function to yield game topics from a file."""
    with open("coco_objects.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 5:
                break
            yield (i, line.strip())


with mlflow.start_run():
        
    def run_experiment(i, topic):

        convo = []

        # Initialize the OpenAI client with the given API key or from environment variables
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize the conversation
        conversation = Conversation(client, topic)
        
        if i == 0:
            conversation.mlflow_log()

        # Start the conversation
        for (host_msg, guesser_msg) in conversation.run_conversation():

            if host_msg:
                print("host_msg: ", host_msg)
            if guesser_msg:
                print("guesser_msg: ", guesser_msg)

            convo.append((host_msg, guesser_msg))



    # Number of workers
    num_workers = 1

    # Using ThreadPoolExecutor to run experiments concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        topics = list(topic())  # Extract topics first to know the total number
        indices, topics = zip(*topics)  # Unzip the list of tuples into two lists
        results = list(tqdm(executor.map(run_experiment, indices, topics), total=len(topics)))

 
