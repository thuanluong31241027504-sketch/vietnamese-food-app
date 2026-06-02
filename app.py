import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import kagglehub
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Cấu hình page
st.set_page_config(
    page_title="Vietnamese Food Recognition 🍜",
    page_icon="🍜",
    layout="wide"
)

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16  # Giảm xuống để tránh memory error
EPOCHS = 10  # Giảm epochs để train nhanh hơn
LEARNING_RATE = 0.001

@st.cache_resource
def load_and_train_model():
    """Load dataset và train model - chỉ chạy 1 lần duy nhất"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Download dataset
    status_text.text("📥 Đang tải dataset từ Kaggle (khoảng 2-3 phút)...")
    progress_bar.progress(10)
    
    try:
        path = kagglehub.dataset_download("quandang/vietnamese-foods")
        progress_bar.progress(20)
        
        # Tạo dataframe
        def create_df(directory):
            filepaths, labels = [], []
            if os.path.exists(directory):
                for label in os.listdir(directory):
                    class_dir = os.path.join(directory, label)
                    if os.path.isdir(class_dir):
                        for file in os.listdir(class_dir):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                filepaths.append(os.path.join(class_dir, file))
                                labels.append(label)
            return pd.DataFrame({'filepath': filepaths, 'label': labels})
        
        # Load data
        train_df = create_df(os.path.join(path, 'Train'))
        valid_df = create_df(os.path.join(path, 'Validate'))
        
        # Nếu không có validation, split từ train
        if len(valid_df) == 0 and len(train_df) > 0:
            train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        progress_bar.progress(30)
        status_text.text(f"✅ Đã tải: {len(train_df)} ảnh train, {len(valid_df)} ảnh validation")
        
    except Exception as e:
        st.error(f"Lỗi tải dataset: {str(e)}")
        return None, None, None
    
    # Step 2: Tạo data generators
    status_text.text("🔄 Đang chuẩn bị data augmentation...")
    progress_bar.progress(40)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    valid_generator = valid_datagen.flow_from_dataframe(
        valid_df,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    class_names = sorted(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    progress_bar.progress(50)
    
    # Step 3: Xây dựng model
    status_text.text("🏗️ Đang xây dựng model CNN...")
    
    model = Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        
        # Block 1
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.30),
        
        # Block 3
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.35),
        
        # Block 4
        Conv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.40),
        
        # Head
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    progress_bar.progress(60)
    
    # Step 4: Train model
    status_text.text(f"🎓 Đang training model với {EPOCHS} epochs (mất 5-10 phút)...")
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    ]
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=callbacks,
        verbose=0
    )
    
    progress_bar.progress(100)
    status_text.text("✅ Training hoàn tất!")
    
    return model, class_names, history

def predict_image(image, model, class_names):
    """Dự đoán ảnh"""
    # Preprocess
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_array = img_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Lấy top 3
    top_indices = np.argsort(predictions)[-3:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'food': class_names[idx],
            'confidence': float(predictions[idx]),
            'percentage': f"{predictions[idx]*100:.1f}%"
        })
    
    return results

# Main UI
st.title("🍜 Vietnamese Food Recognition")
st.markdown("### Nhận diện món ăn Việt Nam bằng trí tuệ nhân tạo")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1999/1999625.png", width=100)
    st.markdown("## 📌 Hướng dẫn")
    st.markdown("""
    1. **Chờ model khởi tạo** (lần đầu 5-10 phút)
    2. **Upload ảnh** món ăn
    3. **Xem kết quả** dự đoán
    """)
    
    st.markdown("## 🍽️ Các món có thể nhận diện")
    st.markdown("""
    - Phở, Bún chả, Bánh mì
    - Cơm tấm, Bánh xèo
    - Chả giò, Gỏi cuốn
    - Và nhiều món khác...
    """)
    
    if st.button("🔄 Train lại model", type="primary"):
        st.cache_resource.clear()
        st.rerun()

# Load model
st.info("⏳ **Lần đầu chạy sẽ mất 5-10 phút để tải dataset và train model. Vui lòng kiên nhẫn!**")

try:
    model, class_names, history = load_and_train_model()
    
    if model is not None:
        st.success("✅ Model đã sẵn sàng!")
        
        # Hiển thị thông tin training
        with st.expander("📊 Xem thông tin training", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Final Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
            with col2:
                st.metric("📈 Best Accuracy", f"{max(history.history['val_accuracy']):.2%}")
            with col3:
                st.metric("🔄 Epochs trained", len(history.history['val_accuracy']))
            
            # Vẽ biểu đồ
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
            ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(history.history['loss'], label='Train', linewidth=2)
            ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # Upload ảnh
        st.markdown("## 📤 Upload ảnh món ăn của bạn")
        
        uploaded_file = st.file_uploader(
            "Chọn file ảnh...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Hỗ trợ các định dạng JPG, PNG, WEBP"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh của bạn", use_column_width=True)
            
            with col2:
                with st.spinner("🔍 Đang phân tích ảnh..."):
                    results = predict_image(image, model, class_names)
                
                st.success("✅ Kết quả dự đoán:")
                
                for i, result in enumerate(results, 1):
                    if i == 1:
                        emoji = "🥇"
                        color = "#FFD700"
                    elif i == 2:
                        emoji = "🥈"
                        color = "#C0C0C0"
                    else:
                        emoji = "🥉"
                        color = "#CD7F32"
                    
                    st.markdown(f"""
                    <div style="
                        padding: 15px;
                        border-radius: 10px;
                        background: linear-gradient(135deg, {color}20, {color}05);
                        margin: 10px 0;
                        border-left: 4px solid {color};
                    ">
                        <h3>{emoji} {result['food']}</h3>
                        <p>Độ tin cậy: <b>{result['percentage']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(result['confidence'], text=f"Confidence: {result['percentage']}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: gray;">
            <p>Powered by TensorFlow & Streamlit | Vietnamese Food Recognition AI</p>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Lỗi: {str(e)}")
    st.markdown("""
    ### Cách khắc phục:
    1. Kiểm tra lại kết nối internet
    2. Refresh trang và thử lại
    3. Nếu vẫn lỗi, đợi vài phút rồi thử lại
    """)

# Thêm thông báo về thời gian chờ
st.info("💡 **Gợi ý:** Lần chạy đầu tiên sẽ mất thời gian để tải dataset (khoảng 500MB) và train model. Các lần sau sẽ nhanh hơn nhờ cache của Streamlit.")
