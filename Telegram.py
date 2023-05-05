import cv2
import telebot
from telebot import types
import numpy as np
from datatoken import TOKEN

# Инициализация бота
bot = telebot.TeleBot(TOKEN)

# Загрузка предобученной YOLOv3 нейросети
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Список допустимых классов для детектирования
classes = []
with open("coco.names", "r") as f:
   classes = [line.strip() for line in f.readlines()]

# Функция для обработки изображения с помощью YOLOv3
def detect_objects(image):
    height, width, _ = image.shape



    # Получаем бинарные данные изображения
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

    # Подаем бинарные данные на вход нейросети
    net.setInput(blob)

    # Получаем выход нейросети
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    # Список обнаруженных объектов
    boxes = []
    confidences = []
    class_ids = []

    # Проходимся по всем выходным слоям и получаем координаты объектов
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Используем алгоритм NMS для удаления дубликатов
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Рисуем рамки вокруг обнаруженных объектов
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(image, f'{label} {int(confidences[i]*100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)




    return image

# Обработка входящих сообщений
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id

    # Загружаем изображение
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    img_file = 'image.jpg'
    with open(img_file, 'wb') as ima:
        ima.write(downloaded_file)

    # Обрабатываем изображение с помощью YOLOv3
    image = cv2.imread(img_file)
    image = detect_objects(image)

    # Отправляем обработанное изображение пользователю
    cv2.imwrite(img_file, image)
    photo = open(img_file, 'rb')
    bot.send_photo(chat_id, photo)

# Запускаем бота
bot.polling()