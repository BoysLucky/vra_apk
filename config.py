import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def mode():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = "cpu"
    model.to(device)

    return model, feature_extractor, tokenizer