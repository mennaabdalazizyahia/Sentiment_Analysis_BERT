import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model 

# Set up the basic page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------------------------------------------------
# 1. Load Assets
# ----------------------------------------------------------------------

# Use @st.cache_resource to prevent model reloading upon user interaction
@st.cache_resource
def load_assets():
    """Loads the models, Label Encoder, and Sentence Encoder."""
    try:
        # 1. Load Sentence Encoder (BERT model)
        sentence_encoder = joblib.load('sentence_encoder.joblib')
        
        # 2. Load the trained Keras/TensorFlow model
        sentiment_model = load_model('sentiment_model.keras')
        
        # 3. Load the Label Encoder
        label_encoder = joblib.load('label_encoder.joblib')

        # Extract class names (Positive, Negative, Neutral)
        classes = label_encoder.classes_
        
        return sentiment_model, sentence_encoder, classes
    
    except FileNotFoundError as e:
        # Clear error message if model files are not found
        st.error(
            f"⚠️ Error: Could not find saved files. Please ensure 'sentiment_model.keras', 'sentence_encoder.joblib', and 'label_encoder.joblib' are in the same folder as 'app.py'."
        )
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        st.stop()

sentiment_model, sentence_encoder, classes = load_assets()

# ----------------------------------------------------------------------
# 2. Prediction Function
# ----------------------------------------------------------------------

def predict_sentiment(text):
    """Converts text to an embedding and predicts sentiment."""
    if not text or not text.strip():
        return None, None
        
    # 1. Encode the text (input must be a list)
    try:
        # show_progress_bar=False is recommended for Streamlit apps
        embedding = sentence_encoder.encode([text], show_progress_bar=False)
    except Exception as e:
        st.error(f"Error encoding text (BERT): {e}")
        return None, None
    
    # 2. Predict using the Keras model
    prediction = sentiment_model.predict(embedding, verbose=0)
    
    # 3. Get the predicted class and confidence
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    sentiment = classes[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    return sentiment, confidence

# ----------------------------------------------------------------------
# 3. Streamlit Interface
# ----------------------------------------------------------------------

st.title("💡 Sentiment Analysis Application")
st.markdown("This application uses a trained **BERT** model (via Sentence Transformers) to classify sentiment into: **Positive**, **Negative**, or **Neutral**.")

# Text input area
user_input = st.text_area(
    "Enter the text to analyze here:",
    placeholder="For example: This product is absolutely wonderful!",
    height=150
)

# Analysis button
if st.button("Analyze Sentiment", type="primary"):
    if user_input:
        
        # Perform prediction
        sentiment, confidence = predict_sentiment(user_input)

        if sentiment:
            # Format results
            confidence_percent = f"{confidence * 100:.2f}%"
            
            # Determine icon and color based on sentiment
            if sentiment == 'Positive':
                icon = "⭐"
                color = "green"
            elif sentiment == 'Negative':
                icon = "❌"
                color = "red"
            else: # Neutral
                icon = "〰️"
                color = "blue"

            # Display results in a stylish card
            st.markdown("---")
            st.subheader(f"Predicted Result: {icon} {sentiment}")
            
            # Use st.columns and st.metric for a clean display
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric(label="Confidence Score", value=confidence_percent)
            
            with col2:
                # Display a progress bar for confidence
                st.progress(confidence)

    else:
        st.warning("Please enter text for analysis before pressing the button.")
