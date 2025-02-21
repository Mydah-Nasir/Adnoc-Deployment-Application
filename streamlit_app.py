import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import gdown
import os
import torch

MODEL_URL = "https://drive.google.com/uc?export=download&id=14iq_-amI-hGSALxfbDDIGcQFMUiFMExC"
MODEL_PATH = "p&id.pt"


if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

# Load the YOLO model
#model = torch.load(MODEL_PATH)

# Load the YOLO model
model = YOLO("p&id.pt")  # Replace with your trained YOLO model

# List of all possible classes
all_classes = [
    "utility_drain_connection", "field_instrument", "pneumatic_line", "main_dcs_console",
    "esd_trip", "gate_value", "interlocked", "centrifugal_motor"
]

# Streamlit UI
st.title("Adnoc P&ID Project")
#st.write("Upload an image, select classes, draw an ROI, and the YOLO model will process it.")

# Sidebar for file upload
st.sidebar.header("Upload Image")
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Sidebar for class selection
st.sidebar.header("Select parts to Display")
selected_classes = st.sidebar.multiselect("Choose parts to display:", all_classes, default=all_classes)

if uploaded_image is not None:
    # Read the image
    image = Image.open(uploaded_image)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize image for better display without cutting edges
    max_width = 800  # Set a maximum display width
    aspect_ratio = image.height / image.width
    display_height = int(max_width * aspect_ratio)
    resized_image = image.resize((max_width, display_height))

    # Display the drawable canvas
    st.write("### Draw a ROI (Region of Interest) on the image")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill
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
            # Get the bounding box coordinates from the drawn rectangle
            obj = objects[0]  # Assuming one ROI
            left, top = int(obj["left"]), int(obj["top"])
            width, height = int(obj["width"]), int(obj["height"])
            right, bottom = left + width, top + height
            
            # Scale coordinates back to original image size
            scale_x = image.width / max_width
            scale_y = image.height / display_height
            left, right = int(left * scale_x), int(right * scale_x)
            top, bottom = int(top * scale_y), int(bottom * scale_y)
            
            # Extract ROI
            roi = img_bgr[top:bottom, left:right]
            
            # Run YOLO inference on ROI
            results = model(roi)
            
            # Count objects detected
            class_counts = {cls: 0 for cls in selected_classes}  # Initialize counts only for selected classes
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
            
            # Annotate ROI
            if filtered_results:
                annotated_frame = filtered_results[0].plot()
            else:
                annotated_frame = results[0].plot()
            
            # Convert counts to a DataFrame and display as a table
            df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
            st.write("### Part Counts in ROI")
            st.table(df)
            
            # Display the annotated ROI image
            st.image(annotated_frame, caption="Annotated ROI", use_container_width=True)
