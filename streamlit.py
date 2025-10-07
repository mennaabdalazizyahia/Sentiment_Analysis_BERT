# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras import models
import joblib
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
import io

# Streamlit page setup
st.set_page_config(
    page_title="Sentiment Analysis with BERT",
    page_icon="📊",
    layout="wide"
)

# App title
st.title("📊 Sentiment Analysis with BERT")
st.markdown("---")

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = models.load_model('sentiment_model.h5')
        le = joblib.load('label_encoder.joblib')
        bert_model = joblib.load('sentence_encoder.joblib')
        return model, le, bert_model
    except:
        st.error("❌ Error loading models. Please make sure all required files exist.")
        return None, None, None

@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
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
        
        return data_training, data_val
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None

def predict_sentiment(text, model, le, bert_model):
    """Predict sentiment for input text"""
    try:
        # Convert text to embedding
        text_embedding = bert_model.encode([text])
        
        # Make prediction
        prediction_probs = model.predict(text_embedding)
        prediction = np.argmax(prediction_probs, axis=1)
        sentiment = le.inverse_transform(prediction)[0]
        confidence = np.max(prediction_probs)
        
        return sentiment, confidence, prediction_probs[0]
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None, None

# Sidebar
st.sidebar.title("🛠️ Settings")
st.sidebar.markdown("---")

# Load models and data
with st.spinner("🔄 Loading models and data..."):
    model, le, bert_model = load_models()
    data_training, data_val = load_data()

if model and le and bert_model and data_training is not None:
    st.sidebar.success("✅ Models loaded successfully!")
    
    # Single prediction section
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
        predict_btn = st.button("🚀 Analyze Sentiment", type="primary")
        st.markdown("---")
        st.markdown("**Examples:**")
        st.markdown("- `I love this product!` → Positive")
        st.markdown("- `The service is terrible` → Negative")
        st.markdown("- `It looks normal` → Neutral")
    
    if predict_btn and user_input:
        with st.spinner("🔄 Analyzing sentiment..."):
            sentiment, confidence, probabilities = predict_sentiment(user_input, model, le, bert_model)
            
            if sentiment:
                st.success(f"✅ Text analyzed successfully!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if sentiment == "Positive":
                        st.metric("Sentiment", "😊 Positive", delta=f"{confidence:.2%}")
                    elif sentiment == "Negative":
                        st.metric("Sentiment", "😠 Negative", delta=f"{confidence:.2%}")
                    else:
                        st.metric("Sentiment", "😐 Neutral", delta=f"{confidence:.2%}")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                with col3:
                    st.metric("Text Length", f"{len(user_input)} chars")
                
                # Sentiment probabilities chart
                st.subheader("📊 Sentiment Probabilities")
                
                sentiments_list = le.classes_
                prob_df = pd.DataFrame({
                    'Sentiment': sentiments_list,
                    'Probability': probabilities
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF4B4B', '#00D4AA', '#1F77B4']  # Red, Green, Blue
                bars = ax.bar(prob_df['Sentiment'], prob_df['Probability'], color=colors, alpha=0.8)
                
                # Add values on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.2%}', ha='center', va='bottom', fontsize=12)
                
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45)
                
                st.pyplot(fig)
    
    st.markdown("---")
    
    # Data analysis section
    st.header("📈 Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Model Evaluation", "🔍 Training Data"])
    
    with tab1:
        st.subheader("Sentiment Distribution in Training Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = data_training['sentiment'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                   colors=['#00D4AA', '#FF4B4B', '#1F77B4'])
            ax1.set_title('Sentiment Distribution in Training Data')
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.bar(sentiment_counts.index, sentiment_counts.values, 
                   color=['#00D4AA', '#FF4B4B', '#1F77B4'])
            ax2.set_title('Sentiment Distribution in Training Data')
            ax2.set_ylabel('Number of Texts')
            plt.xticks(rotation=45)
            st.pyplot(fig2)
    
    with tab2:
        st.subheader("Model Performance on Test Data")
        
        # Calculate predictions for test data
        with st.spinner("🔄 Calculating model performance..."):
            x_val = bert_model.encode(data_val['comments'].tolist(), batch_size=32, show_progress_bar=False)
            y_val_encoded = le.transform(data_val['sentiment'])
            
            y_pred_probs = model.predict(x_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_pred_labels = le.inverse_transform(y_pred)
            y_true_labels = le.inverse_transform(y_val_encoded)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_,
                       cmap='Blues', ax=ax3)
            ax3.set_title('Confusion Matrix')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            st.pyplot(fig3)
        
        with col2:
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Blues'))
    
    with tab3:
        st.subheader("Training Data Sample")
        
        # Data filters
        col1, col2 = st.columns(2)
        with col1:
            selected_sentiment = st.selectbox(
                "Filter by sentiment:",
                ["All"] + list(data_training['sentiment'].unique())
            )
        
        with col2:
            sample_size = st.slider("Sample size:", 5, 50, 10)
        
        # Apply filter
        display_data = data_training
        if selected_sentiment != "All":
            display_data = display_data[display_data['sentiment'] == selected_sentiment]
        
        st.dataframe(display_data.head(sample_size), use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Texts", len(data_training))
        
        with col2:
            st.metric("Training Texts", len(data_training))
        
        with col3:
            st.metric("Test Texts", len(data_val))

else:
    st.error("❌ Failed to load models or data. Please check the required files.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Developed with BERT, TensorFlow, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
