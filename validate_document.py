import cv2
import easyocr  # EasyOCR for better text extraction
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize EasyOCR reader for English
reader = easyocr.Reader(['en'])

# Load the Sentence-BERT model for text similarity
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Setup for BERT-based semantic analysis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to perform OCR on the uploaded document using EasyOCR
def extract_text_from_image(image_path):
    result = reader.readtext(image_path)
    extracted_text = ' '.join([text[1] for text in result])
    return extracted_text

# Layout validation using CNN-based model (ResNet)
def validate_layout(image_path, template_path):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    
    if image is None:
        raise ValueError(f"Failed to load the image at {image_path}")
    if template is None:
        raise ValueError(f"Failed to load the template at {template_path}")

    # Load pretrained ResNet for layout analysis with updated weights parameter
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(image).unsqueeze(0)
    template_tensor = preprocess(template).unsqueeze(0)

    with torch.no_grad():
        image_features = resnet(image_tensor)
        template_features = resnet(template_tensor)
    
    # Calculate cosine similarity between image features
    cos_sim = cosine_similarity(image_features.detach().numpy(), template_features.detach().numpy())
    return float(cos_sim[0][0])  # Return the similarity score (0-1 scale)

# Text similarity validation using BERT (Siamese Network architecture)
def validate_text(extracted_text, reference_text):
    inputs_extracted = tokenizer(extracted_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_reference = tokenizer(reference_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        extracted_embedding = bert_model(**inputs_extracted).last_hidden_state.mean(dim=1)
        reference_embedding = bert_model(**inputs_reference).last_hidden_state.mean(dim=1)
    
    similarity = cosine_similarity(extracted_embedding.detach().numpy(), reference_embedding.detach().numpy())
    return float(similarity[0][0])  # Return the similarity score (0-1 scale)

# Main function to validate the document
def validate_document(document_path, template_path, reference_text):
    try:
        # Step 1: Validate layout using CNN (ResNet)
        layout_confidence = validate_layout(document_path, template_path)
        layout_confidence = round(layout_confidence * 100, 2)  # Round to 2 decimal places
        print(f"Layout Match Confidence: {layout_confidence}%")

        # Step 2: Validate extracted text using EasyOCR
        extracted_text = extract_text_from_image(document_path)
        text_similarity = validate_text(extracted_text, reference_text)
        text_similarity = round(text_similarity * 100, 2)  # Round to 2 decimal places
        print(f"Text Similarity: {text_similarity}%")

        # Step 3: Calculate overall document accuracy
        overall_accuracy = round((layout_confidence * 0.4 + text_similarity * 0.6), 2)  # Adjusted weights and round
        print(f"Overall Document Accuracy: {overall_accuracy}%")

        # Step 4: Document verification result
        if layout_confidence > 85 and text_similarity > 50:
            verification_result = "Document Verified Successfully!"
        else:
            verification_result = "Document Verification Failed!"
        
        # Return the results as a dictionary
        return {
            'layout_confidence': layout_confidence,
            'text_similarity': text_similarity,
            'overall_accuracy': overall_accuracy,
            'verification_result': verification_result
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


