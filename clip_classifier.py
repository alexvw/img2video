import torch
from PIL import Image
from openai import clip
import requests

def get_top_labels(image, num_labels=3):
    # Load the CLIP model and the tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load and preprocess the image
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)

    # Generate a list of candidate labels
    # In this example, we use the WordNet dataset for nouns as candidate labels
    wordnet_url = "https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/ImageNet1000_train.txt"
    labels = [line.split(" ")[1] for line in requests.get(wordnet_url).text.splitlines()]

    # Tokenize the candidate labels
    tokenized_labels = clip.tokenize(labels).to(device)

    # Forward pass
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_image)
        text_features = model.encode_text(tokenized_labels)

    # Calculate the similarity between image and text features
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    scores = similarity[0].cpu().numpy()

    # Get the top labels
    top_indices = scores.argsort()[-num_labels:][::-1]
    top_labels = [(labels[i], scores[i]) for i in top_indices]

    return top_labels

# Load the image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)

# Get the top labels
top_labels = get_top_labels(image)

for label, score in top_labels:
    print(f"{label}: {score:.4f}")