import sys
sys.path.append("")

from sentence_transformers import CrossEncoder
import matplotlib.pyplot as plt 


class Evaluate:

    def __init__(self):
        
        self.model = CrossEncoder("eval/jina-reranker-v1-tiny-en", revision="cb5347e43979c3084a890e3f99491952603ae1b7", trust_remote_code=True)
        
    def _rank(self, query, documents):

        results = self.model.rank(query, documents, return_documents=True)

        results = sorted(results, key=lambda x: x['corpus_id'], reverse=False)

        documents = [result['text'] for result in results]
        scores = [result['score'] for result in results]            

        return documents, scores
    
    def _plot(self, topic, documents, scores):


        # Plotting the scores against the guesses
        plt.figure(figsize=(10, 6))
        plt.plot(documents, scores, marker='o', linestyle='-', color='b')

        # Adding labels and title
        plt.xlabel('Documents')
        plt.ylabel('Score')
        plt.title(f'Scores Tending to the True Value {topic}')

        # Displaying the plot
        plt.grid(True)
        plt.show()


    def evaluate_guesses(self, topic:str, guesses:list):

        documents, scores = self._rank(topic, guesses)

        return documents, scores

    def evaluate_questions(self, topic, questions):

        query = f"Is the secret topic a {topic}?"

        documents, scores = self._rank(query, questions)

        return documents, scores


if __name__ == "__main__":

    eval = Evaluate()

    # Example query and documents
    topic = "Pencil"
    guesses = ['Elephant', 'Cat', 'Chair', 'Table', 'Laptop', 'Pen', 'Pencil']
    guesses, scores = eval.evaluate_guesses(topic, guesses)
    eval._plot(topic, guesses, scores)
    
    questions =  ['Is the secret topic a living organism?', 'Is the secret topic a man-made object?', 'Is the secret topic commonly found inside a household?', 'Is the secret topic used for entertainment or leisure activities?', 'Is the secret topic used for cleaning or maintenance purposes?']
    questions, scores = eval.evaluate_questions(topic, questions)
    eval._plot(topic, questions, scores)



