# Web-приложение для классификации изображений

Web-приложение для классификации изображений. Используются библиотеки:

- [TensorFlow](https://www.tensorflow.org/).
- [Streamlit](https://streamlit.io/).

Для распознавания изображений используется нейронная сеть resnet50 дообученная на дополнительный классах
### Точность распознования 88%

Результаты работы на тестовых даннх в файле: Predict_0.csv

https://github.com/blinov-89/classification_im_Streamlit_avanpost/blob/main/Predict_0.csv

### Для запуска приложения использовать файл: app.py и streamlit

https://github.com/blinov-89/classification_im_Streamlit_avanpost/blob/main/app.py

Предобученная модель CV.H5 хранится по ссылке:

https://drive.google.com/file/d/1Va5C05Howd1ZiKJ0saU0jOgTO29ufGm7/view?usp=share_link

Данные для обучения по ссылке:

https://drive.google.com/drive/folders/12aOszakTkZJFSijLyh4Zsy59R9fudHgt?usp=share_link

Тестовые данные по сслыке:

https://drive.google.com/drive/folders/1sst2k_phARPCI85HUJU9GR8E5w_62C3E?usp=share_link

### Вводные данные: 

Нейронная сеть ResNet50, предобученная на датасете ImageNet.

### Тематика – средства передвижения, перечень исходных классов:

•  трактор
•  газонокосилка
•  велосипед
•  сноуборд
•  лыжи
•  грузовик
•  микроавтобус (газель)
•  поезд
•  самосвал
•  лошадь

### Возможности сервиса:

- Использование предобученной на resnet50 и дообученной на новых классах модели
- Возможность дообучения новыми данными
- Сохранение модели
- Загрузка из сети интернет изображений в нужном количестве нового класса
- Распознавание класса изображений


![image](https://user-images.githubusercontent.com/61515881/205485900-b36652b6-6cc0-4201-b44d-04bfd437a592.png)



