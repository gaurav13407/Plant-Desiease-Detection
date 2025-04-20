import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model:Prediction
def Model_prediction(test_image):
    # Load the model
    model = tf.keras.models.load_model('trained_model.keras')
    image=tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])## Convert Single IMage to a batch
    perediction=model.predict(input_arr)
    result_index=np.argmax(perediction)
    return result_index


## SIdebar
st.sidebar.title("DashBoard")
app_mode=st.sidebar.selectbox("Select Page",("Home","About","Predict"))

## Home Page
if app_mode=="Home":
    st.header("Plant disease prediction")
    image_path="pexels-aditya-aiyar-615049-1407305.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    ## Plant disease prediction using CNN
    This is a simple web application that uses a Convolutional Neural Network (CNN) to predict plant diseases from images.
    The model is trained on a dataset of plant images and can classify them into different categories.
    """)

elif app_mode=="About":
    st.header("About the Project")
    st.markdown("""
    This project is a web application that uses a Convolutional Neural Network (CNN) to predict plant diseases from images.
    The model is trained on a dataset of plant images and can classify them into different categories.
    The application is built using Streamlit and TensorFlow.
    """)

## Prediction Page
elif app_mode=="Predict":
    st.header("Predict Plant Disease")
    st.markdown("""
    Upload an image of a plant leaf to predict its disease.
    The model will classify the image into one of the following categories:
    - Apple Scab
    - Apple Black Rot
    - Apple Cedar Rust
    - Blueberry Mummy Berry
    - Cherry Powdery Mildew
    - Cherry Downy Mildew
    - Corn Cercospora Leaf Spot
    - Corn Common Rust
    - Corn Northern Leaf Blight
    """)
    
    test_image=st.file_uploader("Upload Image",type=["jpg","png","jpeg"])
    if(st.button("Show Image")):
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        ## prediction
    if(st.button("Predict")):
        st.spinner("Processing...")
        st.write("Prediction Result")
        result_index=Model_prediction(test_image)
        ## Define class
        class_names=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success(f"Prediction Result: {class_names[result_index]}")

    
