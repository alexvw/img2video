import torch
from PIL import Image
import clip
import requests
import json
from tqdm import tqdm
import re

def get_top_labels(image, num_labels=3):
    # Load the CLIP model and the tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load and preprocess the image
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)

    # Generate a list of candidate labels
    wordnet_url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    response = requests.get(wordnet_url)

    # Get the response text
    labels_data = response.text

    # Parse the data line by line
    labels_dict = {}
    for line in labels_data.splitlines():
        match = re.match(r'^\s*(\d+):\s*\'(.*)\'\s*,?$', line)
        if match:
            index, labels_str = int(match.group(1)), match.group(2)
            labels_dict[index] = labels_str

    # Flatten the labels into a single list
    labels = [label.strip() for index in labels_dict for label in labels_dict[index].split(',')]

    # Tokenize the candidate labels
    tokenized_labels = clip.tokenize(labels).to(device)

    # Forward pass
    with torch.no_grad():
        # Encode the image
        progress_bar = tqdm(total=1, desc="Encoding image")
        image_features = model.encode_image(preprocessed_image)
        progress_bar.update(1)
        progress_bar.close()

        # Encode the candidate labels
        progress_bar = tqdm(total=1, desc="Encoding text")
        text_features = model.encode_text(tokenized_labels)
        progress_bar.update(1)
        progress_bar.close()

    # Calculate the similarity between image and text features
    progress_bar = tqdm(total=1, desc="Calculating similarity")
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    scores = similarity[0].cpu().numpy()
    progress_bar.update(1)
    progress_bar.close()

    # Get the top labels
    top_indices = scores.argsort()[-num_labels:][::-1]
    top_labels = [(labels[i], scores[i]) for i in top_indices]

    return top_labels

# Load the image
image_path = "input.png"
image = Image.open(image_path)

# Get the top labels
top_labels = get_top_labels(image)

for label, score in top_labels:
    print(f"{label}: {score:.4f}")