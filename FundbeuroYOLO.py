import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Seite konfigurieren
st.set_page_config(page_title="KI Fundbüro (YOLO)", layout="centered")

# Dein YOLO Modell (z.B. trainiert oder yolov8n.pt)
MODEL_FILENAME = "best.pt"  # <- anpassen!

def find_model_path(target_file):
    """Durchsucht das gesamte Projekt nach der Modelldatei."""
    if os.path.exists(target_file):
        return target_file

    for root, dirs, files in os.walk("."):
        if target_file in files:
            return os.path.join(root, target_file)
    return None

@st.cache_resource
def load_model():
    path = find_model_path(MODEL_FILENAME)

    if path:
        try:
            model = YOLO(path)
            return model, path
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei {path}: {e}")
            return None, None
    else:
        st.error(f"Die Datei '{MODEL_FILENAME}' wurde nicht gefunden!")
        st.write("Gefundene Dateien im Ordner:", os.listdir("."))
        return None, None

def predict_image(model, image):
    """Führt YOLO Objekterkennung durch"""
    results = model(image)
    return results

def main():
    st.title("🔍 KI-gestütztes Fundbüro (YOLO)")

    model, path = load_model()

    if path:
        st.success(f"Modell geladen: {path}")

    uploaded_file = st.file_uploader(
        "Bild des Fundstücks hochladen", 
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

        if st.button("Gegenstand erkennen"):
            if model:
                with st.spinner("YOLO analysiert das Bild..."):

                    results = predict_image(model, image)

                    # Ergebnisse visualisieren (mit Bounding Boxes)
                    annotated_image = results[0].plot()

                    st.image(
                        annotated_image, 
                        caption="Erkannte Objekte", 
                        use_column_width=True
                    )

                    # Details anzeigen
                    boxes = results[0].boxes

                    if boxes is not None and len(boxes) > 0:
                        st.subheader("Erkannte Gegenstände:")

                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])

                            label = model.names[cls_id]

                            st.write(f"• {label} ({conf*100:.2f}%)")
                    else:
                        st.write("Keine Objekte erkannt.")
            else:
                st.error("Modell ist nicht bereit.")

if __name__ == "__main__":
    main()
