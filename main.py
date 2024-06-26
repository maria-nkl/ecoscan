import os
import io
import requests
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import time
import pytesseract

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2

from config import PATH_TO_SAVED_MODEL, PATH_TO_LABELS, URI_INFO, URI

def load_model(path_to_saved_model, path_to_labels):
    print('Загружаем модель...')
    start_time = time.time()
    detect_fn = tf.saved_model.load(path_to_saved_model)
    print('Загружаем индексы кодов...')
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Готово... Модель загружена за {} сек.'.format(elapsed_time))
    return detect_fn, category_index

def load_image_into_numpy_array(path):
    image_into_numpy_array = np.array(Image.open(path))
    return image_into_numpy_array

def object_detection(detect_fn, image_np, obj_min_score_thresh, label_min_score_thresh):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=6,
        min_score_thresh=obj_min_score_thresh,
        agnostic_mode=False)

    #Находим координаты рамок
    boxes = detections['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = detections['detection_scores']
    min_score_thresh = label_min_score_thresh
    coordinate_label = {}
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            class_name = category_index[detections['detection_classes'][i]]['name']
            print('На фото найден объект', class_name)
            ymin, xmin, ymax, xmax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            coordinate_label[i] = [class_name, ymin, ymax, xmin, xmax, scores[i]]
    return image_np_with_detections, coordinate_label


def pre_image(image):
    #увеличиваем картинку на 10%
    h, w = image.shape[:2]
    size_up = 0.1
    crop_img = image[int(h/2-w/2*(1-size_up)):int(h/2 + w/2*(1-size_up)), int(w*size_up):int(w*(1-size_up))]

    return crop_img

# подготовка картинки для поиска цифры. Увеличение на 31% (подобрано эмпирическим путем)
def pre_image_number(image):
    h, w = image.shape[:2]
    size_up = 0.31
    crop_img = image[int(h/2-w/2*(1-size_up)):int(h/2 + w/2*(1-size_up)), int(w*size_up):int(w*(1-size_up))]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # перекрашиваем в оттенки серого
    return gray

# поиск цифры внутри рамки # описана в документации
def detect_number(image):
    image = pre_image_number(image)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #вызов приложухи которая ищет цифры
    config = r'--oem 3 --psm 6'
    text_in_image = pytesseract.image_to_string(image, config=config).split('\n') #на выходе строка с распознаными кодами
    labels=set()
    for i in text_in_image:
        if i.isdigit():
            labels.add(i)
    return labels


#Загрузка модели
detect_fn, category_index = load_model(PATH_TO_SAVED_MODEL,PATH_TO_LABELS)

def photo_processing(detect_fn, image_np, min_score_thresh=.4, label_min_score_thresh=.4):
    start_time = time.time()
    cods = list()

    image_np_with_detections, coord_label = object_detection( # большая фотка с обведенным треугольником, коорднаты прямоугольника
        detect_fn,
        image_np,
        min_score_thresh,
        label_min_score_thresh
    )
    img = image_np
    h, w = img.shape[:2]
    finded_number = set()
    #находим абсолютные координаты на картинке
    for number, label in coord_label.items():
        class_name, ymin, ymax, xmin, xmax, scores = label
        scores = int(scores*100)
        ymin = int(ymin * h)
        ymax = int(ymax * h)
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        print('у кода "'+class_name+'" вероятность', scores, '%')

        img1 = img[ymin:ymax, xmin:xmax] #обрезанное фото с треугольником

        finded_number = finded_number.union(detect_number(img1))
    print('!!!!! Найдено', finded_number)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('фото отработано за {} секунды'.format(elapsed_time))
    return finded_number




### TG BOT ###

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types import ContentType, InputFile
import database as db
from config import TOKEN
from crud import receving_inf


bot = Bot(TOKEN)
dp = Dispatcher(bot)

async def on_startup(_):
     await db.db_start()

kb = ReplyKeyboardMarkup(resize_keyboard=True)
b1 = KeyboardButton('/help')
b2 = KeyboardButton('/instruction')
kb.add(b1).add(b2)

HELP_COMMAND = """
<b>/help</b> - <em>Список команд</em>
<b>/instruction</b> - <em>Инструкция по отправке фото</em>
<b>/start</b> - <em>Начать работу с ботом</em>"""

@dp.message_handler(commands=["start"])
async def start_command(message: types.Message) -> None:
    await message.answer(text="Добро пожаловать в наш Телеграмм бот!", reply_markup=kb)
    await message.delete()

@dp.message_handler(commands=["help"])
async def help_command(message: types.Message) -> None:
    await message.reply(text=HELP_COMMAND, parse_mode="HTML")
    await message.delete()

@dp.message_handler(commands=["instruction"])
async def description_command(message: types.Message) -> None:
    await message.answer(text="Для более корректного результата работы, мы советуем вам придерживаться следуюших рекомендаций:\n"
                              "1. Код переработки должен находиться в середине фото\n"
                              "2. Постарайтесь сделать фото так, чтобы оно было качественным и знак переработки находился в резкости\n"
                              "3. Используйте минимальное расстояние от упаковки до камеры Вашего смартфона так, чтобы соблюдались предыдущие пункты\n"
                              "Пример фото:")
    await bot.send_photo(chat_id=message.chat.id, photo=InputFile("instruction.jpg"))
    await message.delete()

@dp.message_handler()
async def correct_input(message: types.Message):
    await message.answer(text="Извините я вас не понимаю, попробуйте ещё раз")

@dp.message_handler(content_types=ContentType.PHOTO)
async def send_image(message: types.Message, detect_fn=detect_fn):
    #достаем фото из телеграмма и записываем ее в переменную img
    file_id = message.photo[-1].file_id
    resp = requests.get(URI_INFO + file_id)
    img_path = resp.json()['result']['file_path']
    img = requests.get(URI+img_path)
    img = Image.open((io.BytesIO(img.content)))

    await message.reply("Началась обработка! Это может занять некоторое время.")

    image_np = np.array(img)
    label_number = set()
    #Последовательно увеличиваем картинку на 10% пять раз. каждый раз делаем распознавание
    for increment in range(5):
        image_np = pre_image(image_np)
        finded_number = photo_processing(detect_fn, image_np, .3, .3)
        if len(label_number) <= len(finded_number):
            label_number = finded_number

    data = receving_inf(label_number)
    label_number = list(label_number)
    s = 0
    if len(data) == 0:
        await message.reply("Адреса не найдены")
    else:
        for center in data:
            if center is None:
                await bot.send_message(message.from_user.id, f"Для кода преработки {label_number[s]} адреса не найдены")
                s += 1
                continue
            s+=1
            await bot.send_message(message.from_user.id, f"Код переработки: {center[0]}\n"
                                                         f"Название центра: {center[1]}\n"
                                                         f"Адрес: {center[2]}\n"
                                                         f"Время работы: {center[3]}\n"
                                                         f"Сотовый телефон: {center[4]}\n")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
