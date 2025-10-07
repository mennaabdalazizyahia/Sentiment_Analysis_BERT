# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras import models, layers
import joblib
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
import time

# Streamlit page setup
st.set_page_config(
    page_title="Sentiment Analysis with BERT - Train & Deploy",
    page_icon="📊",
    layout="wide"
)

# App title
st.title("📊 Sentiment Analysis with BERT - Train & Deploy")
st.markdown("---")

@st.cache_resource
def load_models():
    """Load trained models if they exist"""
    try:
        model = models.load_model('sentiment_model.h5')
        le = joblib.load('label_encoder.joblib')
        bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, le, bert_model
    except:
        st.info("ℹ️ No pre-trained models found. Please train a new model.")
        return None, None, None

@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        with st.spinner("📥 Downloading dataset from Kaggle..."):
            path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
        
        data_training = pd.read_csv(path + '/twitter_training.csv', encoding='utf-8')
        data_val = pd.read_csv(path + '/twitter_validation.csv', encoding='utf-8')
        
        # Standardize column names
        data_training.columns = ['id','entity','sentiment','comments']
        data_val.columns = ['id','entity','sentiment','comments']
        
        # Process data
        data_training = data_training[['sentiment','comments']]
        data_val = data_val[['sentiment','comments']]
        
        sentiment_class = {'Negative': 'Negative','Positive': 'Positive','Neutral':'Neutral','Irrelevant':'Neutral'}
        data_training['sentiment'] = data_training['sentiment'].map(sentiment_class)
        data_val['sentiment'] = data_val['sentiment'].map(sentiment_class)
        
        data_training = data_training.dropna(subset=['comments'])
        
        return data_training, data_val, path
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None

def create_model(architecture_type="Simple"):
    """Create model based on selected architecture"""
    if architecture_type == "Simple":
        model = models.Sequential([
            layers.Input(shape=(384,)),  # BERT embedding size
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(3, activation="softmax")
        ])
    elif architecture_type == "Medium":
        model = models.Sequential([
            layers.Input(shape=(384,)),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(3, activation="softmax")
        ])
    else:  # Complex
        model = models.Sequential([
            layers.Input(shape=(384,)),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(3, activation="softmax")
        ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(data_training, data_val, architecture_type, epochs, batch_size):
    """Train the sentiment analysis model"""
    try:
        # Initialize BERT model
        bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create embeddings
        with st.spinner("🔄 Creating BERT embeddings..."):
            x_train = bert_model.encode(data_training['comments'].tolist(), 
                                      batch_size=32, show_progress_bar=False)
            x_val = bert_model.encode(data_val['comments'].tolist(), 
                                    batch_size=32, show_progress_bar=False)
        
        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(data_training["sentiment"])
        y_val = le.transform(data_val["sentiment"])
        
        # Create and train model
        model = create_model(architecture_type)
        
        # Training progress
        st.subheader("🏋️ Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom callback for progress updates
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f}, Acc: {logs["accuracy"]:.4f}')
        
        # Train model
        with st.spinner("🚀 Training model..."):
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[TrainingCallback()]
            )
        
        # Evaluate model
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
        
        # Save models
        model.save('sentiment_model.h5')
        joblib.dump(le, 'label_encoder.joblib')
        joblib.dump(bert_model, 'sentence_encoder.joblib')
        
        return model, le, bert_model, history, accuracy
        
    except Exception as e:
        st.error(f"❌ Training error: {e}")
        return None, None, None, None, None

def predict_sentiment(text, model, le, bert_model):
    """Predict sentiment for input text"""
    try:
        # Convert text to embedding
        text_embedding = bert_model.encode([text])
        
        # Make prediction
        prediction_probs = model.predict(text_embedding, verbose=0)
        prediction = np.argmax(prediction_probs, axis=1)
        sentiment = le.inverse_transform(prediction)[0]
        confidence = np.max(prediction_probs)
        
        return sentiment, confidence, prediction_probs[0]
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None, None

# Sidebar
st.sidebar.title("🛠️ Model Management")
st.sidebar.markdown("---")

# Load models and data
model, le, bert_model = load_models()
data_training, data_val, data_path = load_data()

# Training Section in Sidebar
st.sidebar.subheader("🎯 Model Training")

if data_training is not None:
    architecture_type = st.sidebar.selectbox(
        "Model Architecture:",
        ["Simple", "Medium", "Complex"],
        help="Simple: 2 layers, Medium: 3 layers, Complex: 4 layers with BatchNorm"
    )
    
    epochs = st.sidebar.slider("Epochs:", 1, 20, 10)
    batch_size = st.sidebar.slider("Batch Size:", 16, 128, 32)
    
    if st.sidebar.button("🚀 Train New Model", type="primary"):
        with st.spinner("Starting model training..."):
            model, le, bert_model, history, accuracy = train_model(
                data_training, data_val, architecture_type, epochs, batch_size
            )
            
        if model is not None:
            st.sidebar.success(f"✅ Training completed! Accuracy: {accuracy:.2%}")
            
            # Show training history
            if history:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history.history['accuracy'], label='Training Accuracy')
                ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                
                ax2.plot(history.history['loss'], label='Training Loss')
                ax2.plot(history.history['val_loss'], label='Validation Loss')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                
                st.pyplot(fig)

# Main content
if model and le and bert_model:
    st.sidebar.success("✅ Models loaded successfully!")
    
    # Prediction section
    st.header("🔍 Analyze New Text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter text to analyze sentiment:",
            placeholder="Type your text here...",
            height=100
        )
    
    with col2:
        st.markdown("### Actions")
        predict_btn = st.button("🎯 Analyze Sentiment", type="primary")
        st.markdown("---")
        st.markdown("**Examples:**")
        st.markdown("- `I love this product!` → Positive")
        st.markdown("- `The service is terrible` → Negative")
    
    if predict_btn and user_input:
        sentiment, confidence, probabilities = predict_sentiment(user_input, model, le, bert_model)
        
        if sentiment:
            st.success(f"✅ **Result:** {sentiment} (Confidence: {confidence:.2%})")
            
            # Display probabilities
            sentiments_list = le.classes_
            prob_df = pd.DataFrame({
                'Sentiment': sentiments_list,
                'Probability': probabilities
            })
            
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#FF4B4B', '#00D4AA', '#1F77B4']
            bars = ax.bar(prob_df['Sentiment'], prob_df['Probability'], color=colors, alpha=0.8)
            
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.2%}', ha='center', va='bottom')
            
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)

else:
    st.info("""
    ## 🚀 Welcome to Sentiment Analysis Trainer!
    
    To get started:
    1. **Configure** your model architecture in the sidebar
    2. **Adjust** training parameters (epochs, batch size)
    3. **Click** "Train New Model" to start training
    4. **Use** the trained model for predictions
    
    No pre-trained models found. Please train a new model using the sidebar options.
    """)

# Data preview section
if data_training is not None:
    st.markdown("---")
    st.header("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", len(data_training))
    with col2:
        st.metric("Validation Samples", len(data_val))
    with col3:
        st.metric("Classes", len(data_training['sentiment'].unique()))
    
    # Show data sample
    if st.checkbox("Show data sample"):
        st.dataframe(data_training.head(10))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Train and deploy your own sentiment analysis models | Built with BERT & TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)
