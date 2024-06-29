from sentence_transformers import CrossEncoder

# Load the model, here we use our tiny sized model
model = CrossEncoder("jinaai/jina-reranker-v1-tiny-en", revision="cb5347e43979c3084a890e3f99491952603ae1b7", trust_remote_code=True)

# Define the path where you want to save the model
model_path = 'eval/jina-reranker-v1-tiny-en'

# Save the model locally
model.save(model_path)