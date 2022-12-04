# %%writefile app.py
import io
import streamlit as st
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import glob
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocessing_function
from keras.models import load_model
from keras.optimizers import Adam
from icrawler.builtin import GoogleImageCrawler

filepath = 'CV.h5'
model = load_model(filepath, compile=True)

mapping = {0: 'велосипед',
           1: 'газонокосилка',
           2: 'грузовик',
           3: 'скейтборд',
           4: 'лошадь',
           5: 'лыжи',
           6: 'микроавтобус(газель)',
           7: 'поезд',
           8: 'самосвал',
           9: 'сноуборд',
           10: 'трактор'}

last_key = list(mapping)[-1]


def save_uploaded_file(uploadedfile):
    if not os.path.exists('photo' + '/' + new_class):
        os.makedirs('photo' + '/' + new_class)

    with open(os.path.join('photo' + '/' + new_class, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Файл {} сохранен в папке {}".format(uploadedfile.name, new_class))


st.title('Новые данные для обучения модели')

new_class = st.text_input('Введите название нового класса')
new_class = new_class.lower()
st.write('Название нового класса: ', new_class)

quabtity = int(st.number_input('Введите количество фото для класса'))

save_cl = st.button('Скачать изображения из интернета')
if save_cl:
    os.makedirs('photo' + '/' + new_class)
    dir_n = 'photo/' + new_class
    google_crawler = GoogleImageCrawler(storage={'root_dir': dir_n})
    google_crawler.crawl(keyword=new_class, max_num=quabtity)
    mapping[last_key + 1] = new_class


PATH_TO_DATA = 'photo'
all_classes = os.listdir(PATH_TO_DATA)

dataset = []
for class_name in all_classes:
    class_images = glob.glob(f"{PATH_TO_DATA}/{class_name}/*")
    class_images = [[img_path, class_name] for img_path in class_images]
    dataset.extend(class_images)
dataset_df = pd.DataFrame(dataset, index=range(len(dataset)), columns=['img_path', 'class'])

train_df, test_df = train_test_split(
    dataset_df,
    test_size=0.2,
    random_state=42)

IMG_SIZE = (224, 224)
config = {
    "batch_size": 64,
    'num_epochs': 15
}
PATH_TO_DATA_new = 'photo'
generator = ImageDataGenerator(preprocessing_function=preprocessing_function)
train_gen = generator.flow_from_dataframe(
    dataframe=train_df,
    directory=PATH_TO_DATA_new,
    x_col='img_path',
    y_col='class',
    class_mode='categorical',
    classes=all_classes,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=config['batch_size'],
    validate_filenames=False
)
test_gen = generator.flow_from_dataframe(
    dataframe=test_df,
    directory=PATH_TO_DATA_new,
    x_col='img_path',
    y_col='class',
    class_mode='categorical',
    classes=train_gen.class_indices.keys(),
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=config['batch_size'],
    validate_filenames=False
)
NUM_STEPS = int(len(train_df) / config['batch_size']) + 1

model_2 = Model(model.input, model.layers[-1].output)
new_model = Sequential()
new_model.add(model_2)
for layer in model_2.layers:
    layer.trainable = False
new_model.add(Dense(len(all_classes), name='new_Dense', activation='softmax'))
opt = Adam(learning_rate=0.001)
new_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


def feature_l(train_gen, test_gen):
    new_model.fit(train_gen, steps_per_epoch=NUM_STEPS, epochs=5, validation_data=test_gen)
    return st.write('Модель обучена')


fit_m = st.button('Обучить модель')
if fit_m:
    history = feature_l(train_gen, test_gen)
    new_model.save('CV.h5')

save_m = st.button('Сохранить модель')
if save_m:
    new_model.save('CV.h5')

st.title('Данные для классификации изображений')


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


# mapping = dict(zip(
#         train_gen.class_indices.values(),
#         train_gen.class_indices.keys()
#     ))
print(mapping)


def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocessing_function(img)
    return img


img = load_image()

result = st.button('Распознать изображение')
if result:
    img_preprocessed = preprocess_image(img)
    st.write('**Результаты распознавания:**')
    prediction = np.argmax(model.predict(img_preprocessed), axis=-1)[0]
    st.write(f'Класс изображения: {prediction}')
    st.write(f'Класс изображения: {mapping[prediction]}')


