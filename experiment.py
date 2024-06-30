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
import logging
from eval.evaluate import Evaluate


logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set level to allow INFO and above

file_handler = logging.FileHandler('experiment.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logging.info("Starting the conversation game...")
logging.warning("This is a warning message.")




# Set the tracking URI to the `mlruns` directory
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name
experiment_name = "Experiment"
mlflow.set_experiment(experiment_name)

def topic():
    """Generator function to yield game topics from a file."""
    with open("coco_objects.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # if i > 2:
            #     break
            yield (i, line.strip())




with mlflow.start_run():

    eval = Evaluate()
        
    def run_experiment(i, topic):

        try:
            logging.info(f"Starting experiment {i} with topic: {topic}")

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
                    logging.info(f"host_msg: {host_msg}")
                if guesser_msg:
                    print("guesser_msg: ", guesser_msg)
                    logging.info(f"guesser_msg: {guesser_msg}")
                convo.append((host_msg, guesser_msg))
                
            try:
                eval.mlflow_log(conversation.host.topic, conversation.guesser_crew.guesses, conversation.guesser_crew.questions)
            except Exception as e:
                print(f"A mlflow error occurred: {e}")
                
            

            

        except Exception as e:
            logging.error(f"An error occurred in experiment {i} with topic: {topic}. Error: {e}")
            print(f"An error occurred in experiment {i} with topic: {topic}. Error: {e}")
          

        logging.info(f"Experiment {i} with topic: {topic} completed.")

        



    # Number of workers
    num_workers = 18

    # Using ThreadPoolExecutor to run experiments concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        topics = list(topic())  # Extract topics first to know the total number
        indices, topics = zip(*topics)  # Unzip the list of tuples into two lists
        results = list(tqdm(executor.map(run_experiment, indices, topics), total=len(topics)))

 
