import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os
from tensorflow.keras.preprocessing import image

# Load models
alz_model = load_model(r'C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\models\alz_classifier.h5')
tumor_model = load_model(r'C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\models\braintumor_new (1).h5')
# pneumonia_model = pickle.load(open(r"C:\GEN AI-PPROJECT\RPC\pneumonia_model.sav", 'rb'))
pneumonia_model=load_model(r'C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\models\pneumonia_model.h5')
############  Welcome Message  ############
print(alz_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']))

import time
st.set_page_config(
    page_title="Intellihealth-MRI-Scanner",
    page_icon=r"C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\images\icon.jpg",
    layout="centered",  # Same as bot code
    initial_sidebar_state="auto",
)
# Define the CSS styles as a string
css = """
<style>
body {
    background-color: '#FF0000';
}

h2 {
    color: rgb(56, 173, 177);
    text-align: center;
}

.custom-paragraph {
    font-size: 18px;
    text-align: center;
    color: #333;
}

.custom-welcome{
    # background-color: #D3D3D3;
    padding: 10px;
    # border: 3px solid black;
    border-radius: 10px;
    margin-top: 20%;
    font-family: 'Comic Sans MS';
}


</style>
"""

# Inject the CSS into the app
st.markdown(css, unsafe_allow_html=True)

# Create a placeholder for the welcome message
welcome_message = st.empty()

if 'msg_displayed' not in st.session_state:          
    # session_state :-  is a special feature in Streamlit that allows you to keep track of variables and their values 
    #                   that persist as users interact with the app, even as the app is reloaded or refreshed.
    st.session_state.msg_displayed = False
    # .msg_displayed :- this creates a variable in the session_state and manage its value.

# Display the customized welcome message using the CSS styles
if not st.session_state.msg_displayed:
    welcome_message.markdown(
        """
        <div class='custom-welcome'>
            
          <h2>IntelliHealth</h2>
               
        </div>
        """,
        unsafe_allow_html=True
    )
    st.session_state.msg_displayed = True


# Wait for 5 seconds
time.sleep(3)

# Clear the welcome message
welcome_message.empty()

####################################

# # Streamlit interface
logo_path=r"C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\images\logo.png"
st.sidebar.image(logo_path, use_column_width=True)
nav_option = st.sidebar.selectbox(
    "Select Operation:",
    ("Alzheimer's Detection", "Brain Tumor Detection", "Pneumonia Detection")
)

# Function to preprocess image for Alzheimer's model
def preprocess_image_for_alzheimer(img):
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

# Function to preprocess image for Tumor model
def preprocess_image_for_tumor(img):
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

# Function to preprocess image for Pneumonia model
def preprocess_image_for_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(120, 120))  # Load and resize the image
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

result=""

def alzheimer():
    global result
    col1, col2 = st.columns([4,1.5])

    with col2:
        memory = Image.open(r'C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\images\m3.jpg')
        st.image(memory,width=250)

    with col1:
        st.title("Alzheimer's Classification")
        st.write("Upload an MRI image to check for Alzheimer's disease.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg", key="alz_uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="RGB", caption="Uploaded MRI Image", use_column_width=True)
        img_array = preprocess_image_for_alzheimer(img)
        prediction = alz_model.predict(img_array)
        index = prediction.argmax()
        if index == 0:
            result="You have Alzheimer's.Its like you have mild Alzheimer's."
            # st.write(result)
        elif index == 1:
            result="You have Alzheimer's.Its like you have moderate Alzheimer's."
            # st.write(result)
        elif index == 2:
            result="You don't have Alzheimer's."
            # st.write(result)
        elif index == 3:
            result="You have Alzheimer's.Its like you have very mild Alzheimer's."
            # st.write(result)
    return st.success(result)

def tumor():
    global result

    col1, col2 = st.columns([5,1])

    with col2:
        brain_tumor = Image.open(r'C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\images\brain.jpg')
        st.image(brain_tumor,width=150)

    with col1:
        st.title("Brain Tumor Classification")
        st.write("Upload an MRI image to check if you have a brain tumor.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg", key="tumor_uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="RGB", caption="Uploaded MRI Image", use_column_width=True)
        img_array = preprocess_image_for_tumor(img)
        prediction = tumor_model.predict(img_array)
        index = prediction.argmax()
        if index == 0:
            result="You have a Tumor.Its you have Glioma Tumor."
            # st.write(result)
        elif index == 1:
            result="You have a Tumor.Its like you have Meningioma Tumor."
            # st.write(result)
        elif index==2:
            result="You do not have a Tumor."
            # st.write(result)

        elif index == 3:
            result="You have a Tumor.Its like you have Pituitary Tumor."
            # st.write(result)
    return st.success(result)    

def pneumonia():
    global result

    col1, col2 = st.columns([5, 1])  # Adjust the ratio of columns as needed

    with col2:
        lungs = Image.open(r'C:\Users\Acer\Desktop\akshat somani\gla akshat somani\projects\IntelliHealth-AI-Health-Care-Assistant-and-X-Ray-Classifier-main\images\lungs1.jpg')
        st.image(lungs, width=150)

    with col1:    
        st.title("Pneumonia Detection")
        st.write("Upload a chest X-ray image to detect Pneumonia.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"], key="pneumonia_uploader")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        temp_file_path = os.path.join("tempDir", uploaded_file.name)
        
        # Process the file (e.g., save it locally)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        preprocessed_image = preprocess_image_for_pneumonia(temp_file_path)

        # col1, col2, col3 = st.columns([3, 1, 3])

        # with col2:
        #     Proceed = st.button('Proceed')

        # if Proceed:
        pneu_prediction = pneumonia_model.predict(preprocessed_image)
        if pneu_prediction[0] > 0.75:
            result='You have pneumonia, which is an infection in your lungs.'
        else:
            result='I’m pleased to inform you that you do not have pneumonia.'
        return  st.success(result)
        
if nav_option == "Brain Tumor Detection":
    tumor()
elif nav_option == "Alzheimer's Detection":
    alzheimer()
elif nav_option == "Pneumonia Detection":
    pneumonia()








