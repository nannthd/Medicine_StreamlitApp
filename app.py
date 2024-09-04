import io
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
from ultralytics import YOLO
from qdrant_client import QdrantClient
import streamlit as st

# Initialize the models and Qdrant client (using your existing setup)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
yolo_model_path = "best_segment.pt"
model_detection = YOLO(yolo_model_path)
model_segmentation = YOLO(yolo_model_path)
qdrant_url = "https://a63ffbf5-5568-46dd-9ec3-98751a51a998.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key = "S0QgrdtYHTC8f_53Nes2uJ4gWoxbPnIwkujhfRlwcWA_MOvuGseLXw"
collection_name = "medicine50classClipModel1.1"
client = QdrantClient(url=qdrant_url, api_key=api_key)

def detect_and_crop_objects(image):
    results = model_detection(image)
    cropped_images = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_images.append(cropped_image)
    return cropped_images

def segment_background(image):
    results = model_segmentation.predict(source=image)
    black_image = np.zeros_like(image)
    for result in results:
        if result.masks is not None:
            masks = result.masks.xy
            for mask in masks:
                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(black_image, [mask], (255, 255, 255))
    binary_mask = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    result_image = cv2.bitwise_and(image, image, mask=binary_mask)
    return result_image

def get_image_vector(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()

def find_top_k_similar_classes_with_qdrant(image, unseenmedicine_folder, k=5, top_n=1000):
    image_vector = get_image_vector(image)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=image_vector.tolist(),
        limit=top_n,
        with_payload=True,
        with_vectors=False
    )
    top_results = [(result.payload.get("classname", "unknown"), result.score) for result in search_result]
    unique_classes = {}
    for class_name, score in top_results:
        if class_name not in unique_classes:
            unique_classes[class_name] = score
        if len(unique_classes) == k:
            break
    filtered_top_k_classes = sorted(unique_classes.items(), key=lambda x: x[1], reverse=True)
    if len(filtered_top_k_classes) < k:
        st.warning("Not enough unique classes found.")
        return
    top_1_score = filtered_top_k_classes[0][1]
    top_2_score = filtered_top_k_classes[1][1] if len(filtered_top_k_classes) > 1 else 0
    if top_1_score > 0.9 and (top_1_score - top_2_score) > 0.02:
        st.success(f"Top prediction: {filtered_top_k_classes[0][0]} with score {top_1_score:.4f}")
    elif top_1_score > 0.9 and (top_1_score - top_2_score) <= 0.02:
        st.error("Please take a clearer picture. >_<")
    elif top_1_score <= 0.9 and top_1_score >= 0.85:
        st.error("Please take a clearer picture. -_-")
    elif top_1_score < 0.85:
        st.error("I have never seen this medicine before.")
        if not os.path.exists(unseenmedicine_folder):
            os.makedirs(unseenmedicine_folder)
        # Save the image only if the condition is met
        # shutil.copy(image_path, unseenmedicine_folder)
        st.info(f"Image saved to {unseenmedicine_folder}")

# Streamlit application layout
st.title("Medicine Image Processing")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the image file directly from memory
    image = Image.open(uploaded_file)
    image = np.array(image.convert("RGB"))
    
    # Automatically start processing when an image is uploaded
    unseenmedicine_folder = r"C:\Users\Admin\Documents\INET\Drug\Medicine_StreamlitApp\unseenmedicine"
    find_top_k_similar_classes_with_qdrant(image, unseenmedicine_folder, k=5)
