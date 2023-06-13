import io

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array


@st.cache_resource()
def load_model():
    return EfficientNetB0(weights='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploaded_file = st.file_uploader('Выберите изображение для распознавания ')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])


def main():
    model = load_model()
    st.title('Распознование изображений в облаке STREAMLIT')
    st.markdown('Проект по программной инженерии')
    st.text('Антропова Н.Г.')

    img = load_image()
    result = st.button('Распознать изображение')
    if result:
        with st.spinner('Подождите...'):
            x = preprocess_image(img)
            preds = model.predict(x)
            st.write('Результат:')
            print_predictions(preds)
        st.success('Готово!')


if __name__ == "__main__":
    main()
