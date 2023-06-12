import io
import pytest
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from streamlit_app import load_model, preprocess_image, print_predictions

def test_load_model():
    model = load_model()
    assert isinstance(model, EfficientNetB0), "Model loading failed, not instance of EfficientNetB0"

def test_preprocess_image():
    # создание тестового изображения
    data = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    processed_image = preprocess_image(img)
    assert processed_image.shape == (1, 224, 224, 3), "Preprocess image function is not working correctly"

def test_print_predictions():
    # для этого теста нам нужно проверить, работает ли функция без ошибок
    # так как мы не можем предсказать конкретный вывод функции
    predictions = np.random.rand(1, 1000)
    try:
        print_predictions(predictions)
        assert True, "Print predictions function works correctly"
    except:
        assert False, "Print predictions function is not working correctly"
