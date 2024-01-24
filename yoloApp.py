import streamlit as st
from PIL import Image, ImageOps
import io
import os
import gdown
#import cv2  # for video processing
#import tempfile  # for handling uploaded files
#from collections import defaultdict
#import numpy as np
import ultralytics

# Define the model loading function outside main
def download_and_load(url):
    trained_model = 'yolov8x_cats.pt'
    if not os.path.exists(trained_model):
        gdown.download(url, trained_model, quiet=False)
    model = ultralytics.YOLO(trained_model)
    return model

def process_image(image, model):
    results = model(image)
    image = ImageOps.exif_transpose(image)
    results = model(image)
    im_array = results[0].plot()
    im = Image.fromarray(im_array[..., ::-1])
    return im


def main():
    
    st.set_page_config(page_title="Interactive Interface for YOLOv8 (prototype)",
                       page_icon="🤖",
                       layout="wide")

    # Initialize session state for login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # User Login
    
    if not st.session_state.logged_in:
        st.sidebar.title("Login")
        username_input = st.sidebar.text_input("Username")
        password_input = st.sidebar.text_input("Password", type="password")

        # Check credentials
        if st.sidebar.button('Login'):
            if username_input == st.secrets['user'] and password_input == st.secrets['password']:
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.sidebar.error("Incorrect username or password")
    else:
    ##############################################################################################################

        model_options = {"Cat Detector": "yolov8x_cats.pt", "YOLO v8 X-Large": "yolov8x.pt", "No Model picked": None}
        st.sidebar.header("Model Selection")

    ##############################################################################################################
        
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model =  list(model_options)[2]

        if 'model' not in st.session_state:
            st.session_state.model = None

    ##############################################################################################################
        
        selected_model = st.sidebar.selectbox("Select a model", list(model_options.keys()), index=list(model_options.keys()).index(st.session_state.selected_model))
        st.title(selected_model)

    ##############################################################################################################

        if selected_model != "No Model picked":
            st.session_state.selected_model = selected_model  
            
            if selected_model == "Cat Detector":
                with st.spinner(f'Loading {selected_model} model...'):
                    url = 'https://drive.google.com/uc?id=116XNjsy9wMEDt76bxJ4q-G5DB5MIal1R'
                    st.session_state.model = download_and_load(url)  
            elif selected_model == "YOLO v8 X-Large":
                with st.spinner(f'Loading {selected_model}...'):
                    st.session_state.model = ultralytics.YOLO(model_options[selected_model])  


            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Check the file format
                #file_type = uploaded_file.type.split('/')[0]

                #if file_type == 'image':
                bytes_data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(bytes_data))
                col1, col2 = st.columns(2)
                with col1:
                    st.image(ImageOps.exif_transpose(image), caption='Image', use_column_width=True)
                with col2:
                    with st.spinner('drawing boxes 🤖🖊️'):
                        processed_image = process_image(image, st.session_state['model'])
                        st.image(processed_image, caption='Detections', use_column_width=True)
                

                #elif file_type == 'video':
                #    # Process video
                #    process_and_track_video(uploaded_file, st.session_state['model'])
                #    st.video(uploaded_file)  # Display the original video

if __name__ == "__main__":
    main()
