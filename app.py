import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from io import BytesIO


# ---------- SET CUSTOM PREMIUM UI ----------
def set_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
        html, body, [class*="css"] {
            background-color: #111111;
            color: #ffffff;
            font-family: 'Orbitron', sans-serif;
        }
        h1 {
            font-size: 3rem !important;
        }
        h2, h3 {
            font-size: 2rem !important;
        }
        .stButton>button {
            border: 3px solid #00ffee;
            color: #00ffee;
            background-color: transparent;
            padding: 15px 35px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 20px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #00ffee;
            color: #111111;
            box-shadow: 0 0 25px #00ffee;
            transform: scale(1.1);
        }
        .stSlider > div > div > div > div {
            background: #00ffee;
            height: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


set_custom_style()

st.title("🚀 AI Art Generator ULTRA PRO™")
st.subheader("The Ultimate Universal AI Style Transfer Studio 🎨🪄")


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


model = load_model()


# ---------- LOAD IMAGE ----------
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    return tf.expand_dims(img, axis=0)


# ---------- PRELOADED STYLE OPTIONS ----------
style_options = {
    "Starry Night": "assets/starry_night.jpg",
    "The Scream": "assets/the_scream.jpg",
    "Mona Lisa": "assets/mona_lisa.jpg",
    "Candy": "assets/candy.jpg"
}

# ---------- USER UPLOAD ----------
content_file = st.file_uploader("📸 Upload Your Content Image", type=['jpg', 'jpeg', 'png'])

use_custom = st.checkbox("🎨 Use my own custom style image (instead of preloaded)")

if use_custom:
    style_file = st.file_uploader("📥 Upload Your Style Image", type=['jpg', 'jpeg', 'png'])
else:
    style_choice = st.selectbox("🎨 Or choose a Preloaded Style Painting", list(style_options.keys()))
    style_file = style_options[style_choice] if content_file else None

if content_file and style_file:
    content_image = load_image(content_file)

    if use_custom:
        style_image = load_image(style_file)
    else:
        style_image = load_image(style_file)

    # ---------- SHOW IMAGES SIDE BY SIDE ----------
    col1, col2 = st.columns(2)
    with col1:
        st.image(np.squeeze(content_image), caption='Your Content Image', width=350)
    with col2:
        st.image(np.squeeze(style_image), caption='Style Image', width=350)

    # ---------- STYLE STRENGTH SLIDER ----------
    strength = st.slider("🎚️ Style Strength", 0.0, 1.0, 1.0, 0.1)

    if st.button("💥 GENERATE AI ARTWORK 💥"):
        with st.spinner('Creating your next-gen masterpiece...'):
            stylized = model(content_image, style_image)[0]
            blended = stylized * strength + content_image * (1 - strength)
            result = tf.clip_by_value(blended, 0.0, 1.0).numpy()
            result_img = Image.fromarray((result[0] * 255).astype(np.uint8))

            st.image(result_img, caption='🔥 Your AI Artwork 🔥', use_column_width=True)

            # ---------- DOWNLOAD BUTTON ----------
            buf = BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="💾 DOWNLOAD ARTWORK",
                               data=byte_im,
                               file_name="AI_Artwork.png",
                               mime="image/png")
else:
    st.info("Upload your content image + style image to start!")
