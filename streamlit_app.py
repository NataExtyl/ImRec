import io
import openai
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image

st.cache_resource()

openai.api_key = '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'


def send_to_openai(preds):
    classes = decode_predictions(preds, top=3)[0]
    predictions = [f"{cl[1]} {cl[2]}" for cl in classes]
    prompt = (
        "Результаты распознавания изображения следующие: "
        + ", ".join(predictions)
        + " Прокомментируй содержание изображения по полученным данным"
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000
    )

    return response.choices[0].text.strip()


def load_model():
    return EfficientNetB0(weights='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    label = 'Выберите изображение для распознавания '
    uploaded_file = st.file_uploader(label=label)
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


model = load_model()
st.title('Распознавание изображений в облаке STREAMLIT')
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
        st.write('Комментарий к изображению:')
        st.write(send_to_openai(preds))

    st.success('Готово!')
