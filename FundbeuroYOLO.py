import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Bild-Categorisierung mit ViT")

# Modell laden (wird gecached, damit es nicht bei jedem Klick neu lädt)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Dein Upload', use_column_width=True)

    # Vorhersage treffen
    with st.spinner('Klassifiziere...'):
        results = classifier(image)

    for res in results:
        st.write(f"**{res['label']}**: {round(res['score'], 4)}")
