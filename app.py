import streamlit as st
import numpy as np
import pickle   
import pandas as pd
import os

# Load trained model
with open('iris_dataset.pkl','rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Iris Prediction App", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Species Prediction")
st.write("Adjust the sliders below to input flower measurements and predict the Iris species.")

# Sliders for user input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Prepare input
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Display user input
st.subheader("📊 Your Input")
input_df = pd.DataFrame(features, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
st.table(input_df)

# Image paths (make sure these files exist in ./images folder)
species_images = {
    "setosa": "images/setosa.jpeg",
    "versicolor": "images/versicolor.jpeg",
    "virginica": "images/virginica.jpg"
}

# Make prediction when button clicked
if st.button("🔍 Predict"):
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]  # probability for each class
    
    species_list = ['setosa', 'versicolor', 'virginica']
    species = species_list[prediction[0]]
    
    st.subheader("🌼 Prediction Result")
    st.success(f"The predicted Iris species is: **{species.capitalize()}**")
    
    # Show image of predicted species if available
    if os.path.exists(species_images[species]):
        st.image(species_images[species], caption=f"Iris {species.capitalize()}", use_column_width=True)
    else:
        st.warning("Image not found for this species.")
    
    # Show probability for each species
    st.subheader("📈 Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Species": species_list,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Species"))

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Iris Dataset ML Model")
