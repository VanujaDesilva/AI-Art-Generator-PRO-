import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from io import BytesIO

# ---------- FINAL COMPANY LAYOUT CSS ----------
def set_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');
        html, body, [class*="css"] {
            background-color: #f9fafb;
            color: #1f2937;
            font-family: 'Orbitron', sans-serif;
        }
        .block-container {
            max-width: 1200px !important;
            padding: 2rem 3rem 3rem 3rem !important;
            margin: auto !important;
        }
        h1 {
            font-size: 2.8rem !important;
            color: #111827;
        }
        h2, h3 {
            font-size: 1.5rem !important;
            color: #111827;
            margin-top: 2rem;
        }
        .stButton>button {
            border: 2px solid #1f2937;
            color: #1f2937;
            background-color: #ffffff;
            padding: 15px 40px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease-in-out;
            margin-top: 1rem;
        }
        .stButton>button:hover {
            background-color: #1f2937;
            color: #ffffff;
            transform: scale(1.05);
        }
        .stSlider > div > div > div > div {
            background: #1f2937;
            height: 8px;
        }
        img {
            border-radius: 15px !important;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.08);
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

# ---------- IMAGE LOADER ----------
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    return tf.expand_dims(img, axis=0)

# ---------- STYLE OPTIONS ----------
style_options = {
    "Starry Night": "assets/starry_night.jpg",
    "The Scream": "assets/the_scream.jpg",
    "Mona Lisa": "assets/mona_lisa.jpg",
    "Candy": "assets/candy.jpg"
}

# ---------- LAYOUT ----------
st.markdown("### Upload Your Content Image")
col1, col2 = st.columns([3, 2])

with col1:
    content_file = st.file_uploader("📸 Upload Image", type=['jpg', 'jpeg', 'png'])

with col2:
    use_custom = st.checkbox("🎨 Use custom style image")
    if use_custom:
        style_file = st.file_uploader("📥 Upload Style Image", type=['jpg', 'jpeg', 'png'])
    else:
        style_choice = st.selectbox("🎨 Or choose a Preloaded Style", list(style_options.keys()))
        style_file = style_options[style_choice] if content_file else None

if content_file and style_file:
    content_image = load_image(content_file)
    style_image = load_image(style_file) if use_custom else load_image(style_file)

    st.markdown("### 🎨 Preview")
    image_col1, image_col2 = st.columns(2)
    with image_col1:
        st.image(np.squeeze(content_image), caption='Your Content Image', use_container_width=True)
    with image_col2:
        st.image(np.squeeze(style_image), caption='Style Image', use_container_width=True)

    strength = st.slider("🎚️ Style Strength", 0.0, 1.0, 1.0, 0.01)

    if st.button("💥 GENERATE AI ARTWORK 💥"):
        with st.spinner('Creating your masterpiece...'):
            stylized = model(content_image, style_image)[0]
            blended = stylized * strength + content_image * (1 - strength)
            result = tf.clip_by_value(blended, 0.0, 1.0).numpy()
            result_img = Image.fromarray((result[0] * 255).astype(np.uint8))

            st.image(result_img, caption='🔥 Your AI Artwork 🔥', use_container_width=True)

            buf = BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="💾 DOWNLOAD ARTWORK",
                               data=byte_im,
                               file_name="AI_Artwork.png",
                               mime="image/png")
else:
    st.info("Upload content + style images to start.")
