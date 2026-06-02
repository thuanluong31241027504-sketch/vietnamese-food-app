import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Import utils
from utils import (
    build_model, compile_model,
    load_and_prepare_data, create_data_generators,
    predict_image, get_top_predictions
)

# Cấu hình trang
st.set_page_config(
    page_title="Vietnamese Food Recognition 🍜",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
IMAGE_SIZE = (128, 128)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
    }
    .css-1d391kg {
        background-color: #F0F2F6;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #F0F2F6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def train_model():
    """Train model với caching"""
    
    with st.status("🚀 Đang khởi tạo...", expanded=True) as status:
        
        # Step 1: Load data
        status.update(label="📥 Đang tải dữ liệu từ Kaggle...")
        train_df, valid_df, test_df = load_and_prepare_data()
        
        if len(train_df) == 0:
            st.error("Không thể tải dữ liệu. Vui lòng kiểm tra kết nối internet!")
            return None, None, None
        
        # Step 2: Create generators
        status.update(label="🔄 Đang chuẩn bị data generators...")
        train_gen, valid_gen, test_gen = create_data_generators(
            train_df, valid_df, test_df, IMAGE_SIZE, BATCH_SIZE
        )
        
        num_classes = len(train_gen.class_indices)
        class_names = sorted(train_gen.class_indices.keys())
        
        # Step 3: Build model
        status.update(label="🏗️ Đang xây dựng model...")
        model = build_model((128, 128, 3), num_classes)
        model = compile_model(model, LEARNING_RATE)
        
        # Step 4: Train model
        status.update(label=f"🎓 Đang training model với {EPOCHS} epochs...")
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=valid_gen,
            callbacks=callbacks,
            verbose=0
        )
        
        status.update(label="✅ Training hoàn tất!", state="complete")
        
        return model, class_names, history

# Main UI
st.title("🍜 Vietnamese Food Recognition")
st.markdown("### Nhận diện các món ăn truyền thống Việt Nam qua ảnh")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1999/1999625.png", width=100)
    st.markdown("## ℹ️ Thông tin")
    st.markdown("""
    **Model Architecture:**
    - CNN từ scratch
    - 4 convolutional blocks
    - BatchNormalization & Dropout
    - Global Average Pooling
    
    **Thông số:**
    - Input size: 128x128
    - Batch size: 32
    - Optimizer: Adam
    - Learning rate: 0.001
    
    **30 món ăn bao gồm:**
    - Phở, Bún chả, Bánh mì
    - Cơm tấm, Bánh xèo, Chả giò
    - Và nhiều món khác...
    """)
    
    st.markdown("---")
    if st.button("🔄 Train lại model từ đầu"):
        st.cache_resource.clear()
        st.rerun()

# Load và train model
try:
    with st.spinner("⏳ Lần đầu chạy sẽ mất 5-10 phút để tải và train model..."):
        model, class_names, history = train_model()
    
    if model is not None:
        # Training metrics
        with st.expander("📊 Training Metrics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Final Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
            with col2:
                st.metric("📈 Best Accuracy", f"{max(history.history['val_accuracy']):.2%}")
            with col3:
                st.metric("📉 Final Loss", f"{history.history['val_loss'][-1]:.3f}")
            with col4:
                st.metric("🔄 Epochs", len(history.history['val_accuracy']))
            
            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
            ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
            ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(history.history['loss'], label='Train', linewidth=2)
            ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
            ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # Upload section
        st.markdown("## 📤 Upload ảnh món ăn")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Chọn file ảnh...",
                type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
                help="Upload ảnh món ăn Việt Nam để nhận diện"
            )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="📸 Ảnh đã upload", use_column_width=True)
            
            with col2:
                with st.spinner("🔍 Đang phân tích ảnh..."):
                    predictions = predict_image(model, image, class_names, IMAGE_SIZE)
                    top_results = get_top_predictions(predictions, class_names, top_k=3)
                
                st.success("✅ Kết quả dự đoán:")
                
                # Hiển thị kết quả với progress bar
                for i, result in enumerate(top_results, 1):
                    confidence = result['confidence'] * 100
                    
                    # Icon cho top 1,2,3
                    if i == 1:
                        icon = "🥇"
                        color = "#FFD700"
                    elif i == 2:
                        icon = "🥈"
                        color = "#C0C0C0"
                    else:
                        icon = "🥉"
                        color = "#CD7F32"
                    
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: {color}10; margin: 10px 0;">
                        <h3>{icon} Top {i}: {result['food']}</h3>
                        <p>Độ tin cậy: <b>{result['percentage']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence / 100)
        
        # Show all food categories
        with st.expander("🍽️ Danh sách các món ăn có thể nhận diện"):
            cols = st.columns(5)
            for idx, food in enumerate(class_names):
                cols[idx % 5].markdown(f"• {food}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: gray;">
        Made with ❤️ using TensorFlow & Streamlit | Vietnamese Food Recognition
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Có lỗi xảy ra: {str(e)}")
    st.markdown("""
    ### Cách khắc phục:
    1. Kiểm tra kết nối internet
    2. Đảm bảo đã cài đặt đúng các thư viện:
    ```bash
    pip install -r requirements.txt
