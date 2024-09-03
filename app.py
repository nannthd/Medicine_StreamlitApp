import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt

# Load your YOLOv8 model
model = YOLO('/content/drive/MyDrive/drug/best_new.pt')

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://8366dca9-9b40-481c-9a7c-102b62b118c2.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="Ft3t6L99uoTsJW8IOk5VH2byKU-BQYqiieuxXFBDp99wo75od0ddAw",
)
collection_name = "vector_CLIP"

# Load CLIPModel and processor for embedding generation
clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs[0].cpu().numpy()

def search_similar_items(query_image, client, collection_name):
    query_embedding = image_embedding(query_image)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=2  # Get top 2 results
    )

    if len(search_result) < 2:
        return None, None

    return search_result[0], search_result[1]

def detect_and_crop(image_path):
    results = model(image_path)
    for result in results:
        boxes = result.boxes.xyxy.cpu().detach().numpy()
        labels = result.boxes.cls.cpu().detach().numpy()
        scores = result.boxes.conf.cpu().detach().numpy()

        threshold = 0.3
        for i, label in enumerate(labels):
            if scores[i] >= threshold:
                x1, y1, x2, y2 = map(int, boxes[i])
                im = Image.open(image_path).convert('RGB')
                im_crop = im.crop((x1, y1, x2, y2))

                return im_crop

    return None

def apply_conditions_and_display(image_path):
    cropped_image = detect_and_crop(image_path)
    if cropped_image is None:
        return

    top_1, top_2 = search_similar_items(cropped_image, qdrant_client, collection_name)
    if top_1 is None or top_2 is None:
        return

    score_1 = top_1.score
    score_2 = top_2.score
    class_name = top_1.payload.get('class', 'Unknown')

    if score_1 > 0.9:
        if (score_1 - score_2) >= 0.02:
            prediction = f"Class Name: {class_name}, Score 1: {score_1:.4f}, Score 2: {score_2:.4f}"
        else:
            prediction = f"Top 1 and top 2 scores are close (Score 1: {score_1:.4f}, Score 2: {score_2:.4f}). Please retake the image."

    elif 0.85 < score_1 < 0.9:
        prediction = f"This drug might be in the class. Score 1: {score_1:.4f}. Please retake the image."

    else:
        prediction = f"This drug is not in the class. Score 1: {score_1:.4f}. Saving as a new class."

    st.image(cropped_image, caption=prediction, use_column_width=True)
    st.write(prediction)

st.title('Drug Classification and Similarity Search')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    apply_conditions_and_display(uploaded_file)
