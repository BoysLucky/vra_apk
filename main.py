import streamlit as st
from PIL import Image
from gtts import gTTS
import torch
from config import mode
import io

device = 'cpu'
model, feature_extractor, tokenizer = mode()

# Text-to-Speech function using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    # Save to an in-memory file
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)  # Reset pointer to the beginning
    # Use Streamlit's built-in audio playback
    st.audio(audio_file, format="audio/mp3")

# Prediction function
def predict_caption(image):
    st.image(image, caption="Captured Image", use_container_width=True)
    
    # Preprocess image
    image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image_tensor, max_length=50)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_text[0]

# Custom CSS for centering content
st.markdown("""
    <style>
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<div class='centered'>", unsafe_allow_html=True)
st.title("Image Captioning with Virtual Relationship Attention (VRA)")
st.write("This app predicts captions for uploaded images or photos taken using your camera.")
st.markdown("</div>", unsafe_allow_html=True)

# File Upload Section
st.markdown("<div class='centered'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

# Camera Input Section
st.markdown("<div class='centered'>", unsafe_allow_html=True)
camera_input = st.camera_input("Take a photo")
st.markdown("</div>", unsafe_allow_html=True)

# Handle image upload
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.write("### Uploaded Image")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("### Predicted Caption")
    try:
        predicted_caption = predict_caption(image)
        st.success(predicted_caption)
        
        # Add TTS Button
        if st.button("🔊 Play Caption"):
            text_to_speech(predicted_caption)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Handle camera input
elif camera_input is not None:
    image = Image.open(camera_input).convert("RGB")
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.write("### Captured Image")
    st.image(image, caption="Captured Image", use_container_width=True)
    
    st.write("### Predicted Caption")
    try:
        predicted_caption = predict_caption(image)
        st.success(predicted_caption)
        
        # Add TTS Button
        if st.button("🔊 Play Caption"):
            text_to_speech(predicted_caption)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.write("Upload an image or take a photo to generate a caption.")
    st.markdown("</div>", unsafe_allow_html=True)
