# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import models, layers
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(
    page_title="Advanced Sentiment Analysis",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Advanced Sentiment Analysis - Train & Predict")
st.markdown("---")

# Initialize session state properly
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'test_accuracy' not in st.session_state:
    st.session_state.test_accuracy = 0.0
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'history' not in st.session_state:
    st.session_state.history = None

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.positive_words = {
            'love', 'like', 'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
            'perfect', 'wonderful', 'outstanding', 'brilliant', 'superb', 'nice', 'best',
            'happy', 'pleased', 'satisfied', 'impressed', 'recommend', 'beautiful', 'fast',
            'easy', 'smooth', 'reliable', 'quality', 'exceeded', 'perfectly', 'working',
            'helpful', 'friendly', 'professional', 'quick', 'affordable', 'valuable', 'enjoy',
            'pleasure', 'delighted', 'marvelous', 'terrific', 'fantastic', 'splendid', 'cool',
            'nice', 'fine', 'decent', 'acceptable', 'positive', 'optimistic', 'hopeful'
        }
        
        self.negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'disappointed',
            'disappointing', 'useless', 'broken', 'waste', 'rubbish', 'garbage', 'trash',
            'slow', 'difficult', 'complicated', 'unreliable', 'cheap', 'expensive', 'overpriced',
            'problem', 'issue', 'error', 'failed', 'crash', 'bug', 'defective', 'faulty',
            'annoying', 'frustrating', 'angry', 'upset', 'regret', 'avoid', 'never', 'dislike',
            'hate', 'loathe', 'despise', 'detest', 'abhor', 'disgusting', 'revolting', 'vile',
            'awful', 'dreadful', 'horrendous', 'appalling', 'shocking', 'unacceptable'
        }
        
        self.intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely', 'totally', 
            'highly', 'incredibly', 'exceptionally', 'particularly', 'especially',
            'remarkably', 'unusually', 'extraordinarily', 'immensely'
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def extract_detailed_features(self, text):
        """Extract comprehensive sentiment features"""
        text = self.preprocess_text(text)
        words = text.split()
        
        # Basic counts
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        
        # Sentiment scores
        total_sentiment_words = positive_count + negative_count
        sentiment_balance = positive_count - negative_count
        sentiment_ratio = positive_count / max(negative_count, 1) if negative_count > 0 else positive_count
        
        # Text characteristics
        text_length = len(text)
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Advanced features
        has_positive = 1 if positive_count > negative_count else 0
        has_negative = 1 if negative_count > positive_count else 0
        is_neutral = 1 if positive_count == 0 and negative_count == 0 else 0
        
        # Compile features
        features = [
            # Basic sentiment counts
            positive_count,
            negative_count,
            sentiment_balance,
            sentiment_ratio,
            
            # Text characteristics
            text_length / 100,
            word_count / 50,
            avg_word_length / 10,
            
            # Advanced indicators
            has_positive,
            has_negative,
            is_neutral,
            intensifier_count,
            
            # Ratios and proportions
            positive_count / max(word_count, 1),
            negative_count / max(word_count, 1),
            total_sentiment_words / max(word_count, 1)
        ]
        
        # Ensure consistent feature length
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def create_training_data(self):
        """Create comprehensive training dataset"""
        
        # Expanded and clearer training data
        positive_samples = [
            "I absolutely love this product it is amazing and perfect",
            "This is excellent quality and works perfectly well",
            "Great value for money and fantastic performance",
            "I am very happy with this purchase it is wonderful",
            "Outstanding product that exceeded my expectations",
            "The quality is superb and delivery was very fast",
            "I am thoroughly impressed with this excellent item",
            "This product is fantastic and works very well",
            "Perfect solution for my needs I love it",
            "Very satisfied with this amazing product quality",
            "This is the best product I have ever bought",
            "Excellent performance and very user friendly",
            "I really like this product it is great",
            "Wonderful experience with this fantastic item",
            "Highly recommended this excellent product",
            "Beautiful design and perfect functionality",
            "Very pleased with this outstanding purchase",
            "This works perfectly and is very reliable",
            "Amazing features and great value for money",
            "I love everything about this perfect product"
        ]
        
        negative_samples = [
            "I hate this terrible product it is awful",
            "This is the worst purchase I have ever made",
            "Poor quality and completely useless product",
            "Very disappointed with this horrible item",
            "This product is garbage and does not work",
            "Terrible quality and waste of my money",
            "I regret buying this awful product",
            "This is completely useless and broken",
            "Very bad experience with this product",
            "Poor performance and terrible quality",
            "This product is defective and unreliable",
            "I am very angry about this purchase",
            "This is absolutely terrible and useless",
            "Worst product ever never buying again",
            "Horrible quality and bad performance",
            "This product is a complete disaster",
            "Very poor construction and cheap materials",
            "I despise this terrible product",
            "This is garbage quality and broken",
            "Extremely disappointed and frustrated"
        ]
        
        neutral_samples = [
            "This product is okay nothing special",
            "It works but is not particularly good",
            "Average quality for the price point",
            "This is a standard product nothing exceptional",
            "It functions normally without issues",
            "Basic features and average performance",
            "This product is acceptable but not great",
            "Normal quality typical for this category",
            "It does the job but could be better",
            "Standard product meets basic requirements",
            "This is fine but not impressive",
            "Average performance and normal quality",
            "It works adequately but nothing special",
            "This product is decent but not amazing",
            "Basic functionality without extra features",
            "Normal product with standard characteristics",
            "It is acceptable for regular use",
            "This meets expectations but does not exceed",
            "Average product with typical performance",
            "It is okay but not remarkable"
        ]
        
        comments = positive_samples + negative_samples + neutral_samples
        sentiments = (['Positive'] * len(positive_samples) + 
                     ['Negative'] * len(negative_samples) + 
                     ['Neutral'] * len(neutral_samples))
        
        return pd.DataFrame({
            'text': comments,
            'sentiment': sentiments
        })
    
    def prepare_features(self, texts):
        """Prepare features for multiple texts"""
        features = []
        for text in texts:
            features.append(self.extract_detailed_features(text))
        return np.array(features)
    
    def create_model(self):
        """Create neural network model"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(20,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, data, epochs=100):
        """Train the sentiment analysis model"""
        # Prepare features and labels
        X = self.prepare_features(data['text'].tolist())
        le = LabelEncoder()
        y = le.fit_transform(data['sentiment'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train model
        model = self.create_model()
        
        # Train with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        return model, le, history, test_accuracy, X_test, y_test
    
    def predict_sentiment(self, text, model, le):
        """Predict sentiment for a single text"""
        features = self.prepare_features([text])
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        sentiment = le.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction[0])
        
        # Get feature analysis
        feature_analysis = self.analyze_features(text)
        
        return sentiment, confidence, prediction[0], feature_analysis
    
    def analyze_features(self, text):
        """Analyze features for display"""
        features = self.extract_detailed_features(text)
        words = self.preprocess_text(text).split()
        
        positive_found = [word for word in words if word in self.positive_words]
        negative_found = [word for word in words if word in self.negative_words]
        intensifiers_found = [word for word in words if word in self.intensifiers]
        
        return {
            'positive_words': positive_found,
            'negative_words': negative_found,
            'intensifiers': intensifiers_found,
            'positive_count': len(positive_found),
            'negative_count': len(negative_found),
            'sentiment_balance': len(positive_found) - len(negative_found)
        }

# Initialize analyzer
analyzer = AdvancedSentimentAnalyzer()

# Sidebar
st.sidebar.title("⚙️ Model Training")
st.sidebar.markdown("---")

# Training section
st.sidebar.subheader("Train Model")
epochs = st.sidebar.slider("Training Epochs", 50, 200, 100)

if st.sidebar.button("🚀 Train Advanced Model", type="primary", use_container_width=True):
    with st.spinner("Training advanced sentiment model... This may take a few seconds"):
        # Create training data
        training_data = analyzer.create_training_data()
        
        # Train model
        model, le, history, test_accuracy, X_test, y_test = analyzer.train_model(
            training_data, epochs
        )
        
        # Store in session state
        st.session_state.model = model
        st.session_state.le = le
        st.session_state.model_trained = True
        st.session_state.training_data = training_data
        st.session_state.test_accuracy = test_accuracy
        st.session_state.history = history
        
        st.sidebar.success(f"✅ Model trained! Accuracy: {test_accuracy:.2%}")

# Main content
if st.session_state.model_trained and st.session_state.model is not None:
    # Show accuracy only if it exists
    if hasattr(st.session_state, 'test_accuracy') and st.session_state.test_accuracy > 0:
        st.sidebar.metric("Test Accuracy", f"{st.session_state.test_accuracy:.2%}")
    
    # Prediction section
    st.header("🔍 Analyze Text Sentiment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = st.text_area(
            "Enter your text:",
            height=120,
            placeholder="Type your text here to analyze sentiment...",
            value="I really love this amazing product! It's perfect!"
        )
    
    with col2:
        st.markdown("### Actions")
        analyze_btn = st.button("🎯 Analyze Sentiment", type="primary", use_container_width=True)
        st.markdown("---")
        st.markdown("**Test Examples:**")
        st.markdown("- `I love this amazing product!`")
        st.markdown("- `This is terrible and useless`") 
        st.markdown("- `It's okay but not great`")
    
    if analyze_btn and user_text:
        with st.spinner("Analyzing sentiment with advanced features..."):
            sentiment, confidence, probabilities, feature_analysis = analyzer.predict_sentiment(
                user_text, st.session_state.model, st.session_state.le
            )
        
        # Display results
        st.success(f"✅ **Prediction: {sentiment}**")
        
        # Results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if sentiment == "Positive":
                st.metric("Sentiment", "😊 Positive", delta=f"{confidence:.1%}")
            elif sentiment == "Negative":
                st.metric("Sentiment", "😠 Negative", delta=f"{confidence:.1%}")
            else:
                st.metric("Sentiment", "😐 Neutral", delta=f"{confidence:.1%}")
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.metric("Text Length", f"{len(user_text)}")
        
        # Feature analysis
        st.subheader("🔍 Feature Analysis")
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            st.metric("Positive Words", feature_analysis['positive_count'])
            if feature_analysis['positive_words']:
                st.write("Found:", ", ".join(feature_analysis['positive_words']))
        
        with feat_col2:
            st.metric("Negative Words", feature_analysis['negative_count'])
            if feature_analysis['negative_words']:
                st.write("Found:", ", ".join(feature_analysis['negative_words']))
        
        with feat_col3:
            st.metric("Sentiment Balance", feature_analysis['sentiment_balance'])
            if feature_analysis['intensifiers']:
                st.write("Intensifiers:", ", ".join(feature_analysis['intensifiers']))
        
        # Probabilities chart
        st.subheader("📊 Sentiment Probabilities")
        
        sentiment_classes = st.session_state.le.classes_
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#FF4B4B', '#1F77B4', '#00D4AA']
        bars = ax.bar(sentiment_classes, probabilities, color=colors, alpha=0.8)
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title('Sentiment Confidence Distribution')
        
        st.pyplot(fig)
    
    # Training results section
    st.markdown("---")
    st.header("📈 Training Results")
    
    tab1, tab2, tab3 = st.tabs(["Training History", "Training Data", "Model Info"])
    
    with tab1:
        st.subheader("Model Training Progress")
        
        if st.session_state.history is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Accuracy plot
            ax1.plot(st.session_state.history.history['accuracy'], label='Training Accuracy')
            ax1.plot(st.session_state.history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss plot
            ax2.plot(st.session_state.history.history['loss'], label='Training Loss')
            ax2.plot(st.session_state.history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        else:
            st.info("No training history available")
    
    with tab2:
        st.subheader("Training Data Sample")
        if st.session_state.training_data is not None:
            st.dataframe(st.session_state.training_data, use_container_width=True)
            
            # Data statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(st.session_state.training_data))
            with col2:
                counts = st.session_state.training_data['sentiment'].value_counts()
                st.metric("Positive Samples", counts['Positive'])
            with col3:
                st.metric("Negative Samples", counts['Negative'])
        else:
            st.info("No training data available")
    
    with tab3:
        st.subheader("Model Architecture")
        st.code("""
        Sequential([
            Dense(128, activation='relu', input_shape=(20,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])
        """)

else:
    # Welcome screen
    st.header("🚀 Advanced Sentiment Analysis")
    
    st.info("""
    ### 📖 How to get started:
    
    1. **Click 'Train Advanced Model'** in the sidebar
    2. **Wait for training to complete** (10-30 seconds)
    3. **Enter your text** and click 'Analyze Sentiment'
    4. **View detailed analysis** with feature breakdown
    
    ### 🎯 Features:
    - **Advanced feature extraction** from text
    - **Comprehensive training data** with clear patterns
    - **Real-time sentiment analysis** with confidence scores
    - **Feature breakdown** showing why each prediction was made
    - **High accuracy** for clear positive/negative statements
    
    ### 🔧 Ready to start?
    **Go to the sidebar and train the model!**
    """)
    
    # Show sample training data
    sample_data = analyzer.create_training_data()
    st.subheader("📋 Sample Training Data")
    st.dataframe(sample_data.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Advanced Sentiment Analysis | Real Training & Accurate Predictions
    </div>
    """,
    unsafe_allow_html=True
)
