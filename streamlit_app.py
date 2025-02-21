import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import gdown
import os
import fitz  # PyMuPDF
import io

# Function to convert PDF pages to images
def pdf_to_images(pdf_bytes):
    """Convert PDF pages to images using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Read PDF from memory
    images = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# Download model if not present
MODEL_URL = "https://drive.google.com/uc?export=download&id=14iq_-amI-hGSALxfbDDIGcQFMUiFMExC"
MODEL_PATH = "p&id.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

# Load YOLO model
model = YOLO(MODEL_PATH)

# List of all possible classes
all_classes = [
    "utility_drain_connection", "field_instrument", "pneumatic_line", "main_dcs_console",
    "esd_trip", "gate_value", "interlocked", "centrifugal_motor"
]

# Streamlit UI
st.title("Adnoc P&ID Project")

# Sidebar for file upload
st.sidebar.header("Upload Image or PDF")
uploaded_file = st.sidebar.file_uploader("Choose an image or PDF...", type=["jpg", "png", "jpeg", "pdf"])

# Sidebar for class selection
st.sidebar.header("Select parts to Display")
selected_classes = st.sidebar.multiselect("Choose parts to display:", all_classes, default=all_classes)

image = None  # Initialize image variable

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        pdf_bytes = uploaded_file.read()  # Read the uploaded file as bytes
        images = pdf_to_images(pdf_bytes)
        selected_image = st.sidebar.selectbox("Select Page", range(1, len(images) + 1))
        image = images[selected_image - 1]
    else:
        image = Image.open(uploaded_file)

if image is not None:
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize image for better display
    max_width = 800
    aspect_ratio = image.height / image.width
    display_height = int(max_width * aspect_ratio)
    resized_image = image.resize((max_width, display_height))

    # Display drawable canvas
    st.write("### Draw a ROI (Region of Interest) on the image")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=resized_image,
        update_streamlit=True,
        height=display_height,
        width=max_width,
        drawing_mode="rect",
        key="canvas"
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[0]
            left, top = int(obj["left"]), int(obj["top"])
            width, height = int(obj["width"]), int(obj["height"])
            right, bottom = left + width, top + height

            # Scale back ROI to original image size
            scale_x = image.width / max_width
            scale_y = image.height / display_height
            left, right = int(left * scale_x), int(right * scale_x)
            top, bottom = int(top * scale_y), int(bottom * scale_y)

            roi = img_bgr[top:bottom, left:right]  # Crop ROI

            # Run YOLO model on ROI
            results = model(roi)

            # Filter results based on selected classes
            class_counts = {cls: 0 for cls in selected_classes}
            filtered_results = []

            for result in results:
                filtered_boxes = []
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name in selected_classes:
                        class_counts[class_name] += 1
                        filtered_boxes.append(box)
                if filtered_boxes:
                    result.boxes = filtered_boxes
                    filtered_results.append(result)

            if filtered_results:
                annotated_frame = filtered_results[0].plot()
            else:
                annotated_frame = results[0].plot()

            # Display class counts as a table
            df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
            st.write("### Part Counts in ROI")
            st.table(df)

            # Display annotated image
            st.image(annotated_frame, caption="Annotated ROI",use_container_width=True)
