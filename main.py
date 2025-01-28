import streamlit as st
from PIL import Image
from gtts import gTTS
import torch
from config import mode
import io
import base64

# Device setup
device = 'cpu'
model, feature_extractor, tokenizer = mode()

# Audio Playback with Autoplay
def autoplay_audio3(file, autoplay=True):
    """Plays the audio file with autoplay option."""
    b64 = base64.b64encode(file).decode()
    if autoplay:
        md = f"""
            <audio id="audioTag" controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg" format="audio/mpeg">
            </audio>
        """
    else:
        md = f"""
            <audio id="audioTag" controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg" format="audio/mpeg">
            </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# Text-to-Speech function using gTTS
def text_to_speech(text):
    """Generates speech from text using gTTS."""
    tts = gTTS(text=text, lang='en')
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file.read()

# Caption Prediction
def predict_caption(image):
    """Predicts the caption for the uploaded image."""
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
        
        # Generate TTS audio and autoplay
        audio_bytes = text_to_speech(predicted_caption)
        autoplay_audio3(audio_bytes)
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
        
        # Generate TTS audio and autoplay
        audio_bytes = text_to_speech(predicted_caption)
        autoplay_audio3(audio_bytes)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.write("Upload an image or take a photo to generate a caption.")
    st.markdown("</div>", unsafe_allow_html=True)
