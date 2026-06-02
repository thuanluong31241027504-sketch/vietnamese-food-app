import numpy as np
from PIL import Image

def predict_image(model, image, class_names, img_size=(128, 128)):
    """Dự đoán món ăn từ ảnh upload"""
    # Resize và preprocessing
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    
    # Chuẩn hóa shape
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_array = img_array.reshape(1, img_size[0], img_size[1], 3)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    
    return predictions

def get_top_predictions(predictions, class_names, top_k=3):
    """Lấy top K dự đoán với độ tin cậy cao nhất"""
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'food': class_names[idx],
            'confidence': float(predictions[idx]),
            'percentage': f"{predictions[idx]*100:.1f}%"
        })
    
    return results
