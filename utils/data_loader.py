import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_dataframe(directory):
    """Tạo dataframe từ thư mục ảnh"""
    filepaths, labels = [], []
    
    if not os.path.exists(directory):
        return pd.DataFrame(columns=['filepath', 'label'])
    
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(class_dir, file))
                    labels.append(label)
    
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

def load_and_prepare_data():
    """Tải dữ liệu từ Kaggle và chuẩn bị dataframe"""
    print("Đang tải dữ liệu từ Kaggle...")
    path = kagglehub.dataset_download("quandang/vietnamese-foods")
    
    # Tạo dataframe cho từng tập
    train_df = create_dataframe(os.path.join(path, 'Train'))
    valid_df = create_dataframe(os.path.join(path, 'Validate'))
    test_df = create_dataframe(os.path.join(path, 'Test'))
    
    # Nếu không có validation set, split từ train
    if len(valid_df) == 0 and len(train_df) > 0:
        train_df, valid_df = train_test_split(
            train_df, test_size=0.2, random_state=42
        )
    
    print(f"Train: {len(train_df)} | Validation: {len(valid_df)} | Test: {len(test_df)}")
    
    return train_df, valid_df, test_df

def create_data_generators(train_df, valid_df, test_df, img_size=(128, 128), batch_size=32):
    """Tạo data generators với augmentation"""
    
    # Data augmentation cho training
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
    
    # Chỉ rescale cho validation và test
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Tạo generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    valid_generator = valid_test_datagen.flow_from_dataframe(
        valid_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = valid_test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, valid_generator, test_generator
