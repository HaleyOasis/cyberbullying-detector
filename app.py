import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown
import os
import zipfile

# Page config
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the merged model"""
    if not os.path.exists("./merged_model"):
        with st.spinner("Downloading model... This may take 2-3 minutes"):
            # Replace with your actual Google Drive file ID
            file_id = "1GyKtvsqboW6UG9WaEa3tqz4HhRz0w2Of"  # Replace this!
            
            try:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", "./model.zip")
                
                with zipfile.ZipFile("./model.zip", 'r') as zip_ref:
                    zip_ref.extractall("./merged_model")
                os.remove("./model.zip")
                
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
    
    tokenizer = AutoTokenizer.from_pretrained("./merged_model")
    model = AutoModelForSequenceClassification.from_pretrained("./merged_model")
    model.eval()
    
    return tokenizer, model

def predict(text, tokenizer, model):
    """Predict cyberbullying"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = predictions[0].tolist()
    
    return {
        'normal': probs[0],
        'cyberbullying': probs[1]
    }

# Main app
st.title("🛡️ Cyberbullying Detection System")
st.markdown("AI-powered detection of cyberbullying and online harassment")

# Load model
tokenizer, model = load_model()
st.success("Model loaded!")

# Input
text_input = st.text_area(
    "Enter text to analyze:",
    placeholder="Type your text here to check for cyberbullying...",
    height=100
)

# Analyze button
if st.button("🔍 Analyze Text", type="primary"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            result = predict(text_input, tokenizer, model)
            
            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("😊 Normal Text", f"{result['normal']:.1%}")
            with col2:
                st.metric("⚠️ Cyberbullying", f"{result['cyberbullying']:.1%}")
            
            # Conclusion
            if result['cyberbullying'] > result['normal']:
                st.error("⚠️ Potential cyberbullying detected")
                st.warning("This text may contain harmful or harassing content")
            else:
                st.success("✅ Text appears normal")
                st.info("No cyberbullying patterns detected")
    else:
        st.warning("Please enter some text to analyze")

# Examples
st.subheader("💡 Try Examples:")
examples = [
    "Great job on your presentation today!",
    "Nobody likes you, you should just disappear",
    "Thanks for helping me with homework",
    "You're so ugly and stupid, kill yourself"
]

cols = st.columns(2)
for i, example in enumerate(examples):
    with cols[i % 2]:
        if st.button(f"Example {i+1}", key=f"ex_{i}", help=f'"{example}"'):
            st.text_area("Enter text to analyze:", value=example, key="example_text")

# Info section
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    **What is Cyberbullying?**
    - Online harassment or intimidation
    - Repeated aggressive behavior
    - Threats, insults, or exclusion
    - Content meant to hurt or embarrass
    
    **How it works:**
    - Uses Fine-tuned HateBERT model
    - Trained specifically on cyberbullying datasets
    - Analyzes text patterns and context
    - Provides confidence scores
    """)

# Footer
st.markdown("---")
st.markdown("🔬 Built with Fine-tuned HateBERT | 🎯 Specialized for Cyberbullying Detection")
