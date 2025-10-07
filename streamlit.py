import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go

st.set_page_config(
    page_title='Sentiment Analysis',
    initial_sidebar_state='expanded'
)

@st.cache_resource
def load_models():
    try:
        # جرب تحمل الموديل بطرق مختلفة
        model = None
        
        # طريقة 1: حمل الموديل مع تجاهل الـ custom objects
        try:
            model = tf.keras.models.load_model('sentiment_model.keras', compile=False)
            st.success("✅ Model loaded from .keras file")
        except Exception as e1:
            st.warning(f"⚠️ Failed to load .keras: {str(e1)[:100]}")
            
            # طريقة 2: حمل من .h5
            try:
                model = tf.keras.models.load_model('sentiment_model.h5', compile=False)
                st.success("✅ Model loaded from .h5 file")
            except Exception as e2:
                st.warning(f"⚠️ Failed to load .h5: {str(e2)[:100]}")
                
                # طريقة 3: بناء الموديل من جديد وتحميل الـ weights
                st.info("🔧 Attempting to rebuild model and load weights...")
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(256, activation="relu", input_shape=(384,)),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(3, activation="softmax")
                ])
                
                try:
                    model.load_weights('model_weights.h5')
                    st.success("✅ Model rebuilt and weights loaded")
                except Exception as e3:
                    st.error(f"❌ All loading methods failed: {str(e3)[:100]}")
                    return None, None, None
        
        # إعادة compile الموديل
        if model is not None:
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # حمل الـ label encoder
        le = joblib.load('label_encoder.joblib')
        
        # حمل الـ BERT model
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return model, le, bert_model
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

def encode_text(text, bert_model):
    """تحويل النص لـ embedding باستخدام BERT"""
    embeddings = bert_model.encode([text])
    return embeddings

def predict_sentiment(text, model, le, bert_model):
    """توقع المشاعر من النص"""
    # Get embedding
    embedding = encode_text(text, bert_model)
    
    # Predict using Keras model
    probabilities = model.predict(embedding, verbose=0)[0]
    
    # Get predicted class
    predicted_class = np.argmax(probabilities)
    predicted_label = le.inverse_transform([predicted_class])[0]
    confidence = probabilities[predicted_class]
    
    return predicted_label, confidence, probabilities

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #c3e6cb;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #f5c6cb;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #ffeaa7;
    }
    </style>
    """, unsafe_allow_html=True)            
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis</h1>', unsafe_allow_html=True)

    model, le, bert_model = load_models()
    
    if model is None:
        st.error("❌ Could not load model. Please check the troubleshooting steps below:")
        with st.expander("🔧 Troubleshooting Steps"):
            st.markdown("""
            **The model file has compatibility issues. Please:**
            
            1. **Re-save your model** using this code in your training script:
            ```python
            # Instead of:
            # model = models.Sequential([
            #     layers.Input(shape=(384,)),
            #     layers.Dense(256, activation="relu"),
            #     ...
            # ])
            
            # Use this:
            model = models.Sequential([
                layers.Dense(256, activation="relu", input_shape=(384,)),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(3, activation="softmax")
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Train your model...
            # history = model.fit(...)
            
            # Save correctly:
            model.save('sentiment_model_fixed.keras')
            ```
            
            2. Replace the old `sentiment_model.keras` with the new `sentiment_model_fixed.keras`
            
            3. Redeploy your app
            """)
        return
    
    st.subheader("✍️ Write a Comment:")
    input_option = st.radio("Choose input method:", 
                          ["Write a comment", "Sample text entry"])
    
    if input_option == 'Write a comment':
        user_input = st.text_area(
            "Write here:",
            placeholder="EX: I love this product! It's brilliant!",
            height=150
        )  
    else:
        sample_texts = {
            "Positive 😊": "I absolutely love this! It's fantastic and amazing!",
            "Negative 😠": "This is terrible and awful. I hate it so much!",
            "Neutral 😐": "The product is okay, nothing special but not bad either."
        }
        selected_sample = st.selectbox("Select a sample", list(sample_texts.keys()))
        user_input = sample_texts[selected_sample]
        st.text_area("Selected text:", user_input, height=100)
    
    if st.button('🔍 Analyze Sentiment', use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    predicted_label, confidence, probabilities = predict_sentiment(
                        user_input, model, le, bert_model
                    )
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    return
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader('📊 Results:')
                
                sentiment_emojis = {
                    "Positive": "😊 Positive",
                    "Negative": "😠 Negative", 
                    "Neutral": "😐 Neutral"
                }
                sentiment_classes = {
                    "Positive": "sentiment-positive",
                    "Negative": "sentiment-negative",
                    "Neutral": "sentiment-neutral"
                }
                
                display_label = sentiment_emojis.get(predicted_label, predicted_label)
                css_class = sentiment_classes.get(predicted_label, "sentiment-neutral")
                
                st.markdown(f'<div class="{css_class}"><h2 style="text-align:center; margin:0;">{display_label}</h2></div>', 
                          unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Confidence Level", f"{confidence:.2%}")
                st.progress(float(confidence))
            
            with col2:
                st.subheader("📈 Probability Distribution")
                
                labels = le.classes_
                colors = []
                for label in labels:
                    if label == "Positive":
                        colors.append('#2ecc71')
                    elif label == "Negative":
                        colors.append('#e74c3c')
                    else:
                        colors.append('#f39c12')
                
                fig = go.Figure(data=[
                    go.Bar(x=labels, y=probabilities,
                          marker_color=colors,
                          text=[f'{p:.1%}' for p in probabilities],
                          textposition='auto')
                ])
                
                fig.update_layout(
                    title="Sentiment Probabilities",
                    xaxis_title="Sentiment",
                    yaxis_title="Probability",
                    yaxis=dict(tickformat='.0%'),
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please input a text to analyze.")

if __name__ == "__main__":
    main()
