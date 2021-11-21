import base64
import yaml
import colorsys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageDraw import ImageDraw
import streamlit as st

from requests_toolbelt import MultipartEncoder
import requests


COLOR_STATE_MAP = {
    'healthy': (0, 200, 250),
    'angular_leafspot': (200, 200, 200),
    'leaf_scorch': (200, 200, 200),
    'leaf_spot': (200, 200, 200),
    'powdery_mildew': (200, 200, 200),
}
MAP_LEAVES_STATE = {
    'healthy': 'Здоровый',
    'angular_leafspot': 'Ожог листьев',
    'leaf_scorch': 'Пятнистость на краях',
    'leaf_spot': 'Пятнистость',
    'powdery_mildew': 'Мучнистая роса'
}
MAP_GROWTH_PHASE = {
    'seed': 'Начало роста',
    'first_leaves': 'Появления первых листьев',
    'pre - flower': 'Начало цветения',
    'flower': 'Цветение',
    'flowers_falling': 'Опадение лепестков',
    'pre - berry': 'Формирование ягод',
    'berry': 'Развитие ягод',
    'mustaches': 'Появление усов',
}
COLOR_FLOWER = (0, 10, 200)
MASK_COLORS = {
    'berry': (10, 10, 250),
    'flower': COLOR_FLOWER,
    'leaf': COLOR_STATE_MAP['healthy']
}


BBOX_WIDTH = 4


def read_config():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    return config


def draw_bounding_box_on_image(
        draw: ImageDraw, x1, y1, x2, y2, color, width=4, display_str_list=()
):
    draw.rounded_rectangle((x1, y1, x2, y2), fill=None,
                           outline=color,
                           width=width,
                           radius=8)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = y1
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.1 * text_height)
        draw.rounded_rectangle(
            [
                (x1, text_bottom - text_height - margin),
                (x1 + text_width + 2 * margin, text_bottom)
            ],
            fill=color,
            radius=5,
        )
        draw.text(
            (x1 + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= (text_height + 2 * margin)


def get_mask(b64mask, width, height) -> np.ndarray:
    mask = np.frombuffer(
        base64.decodebytes(b64mask.encode('utf-8')), dtype=np.uint8
    )
    mask = mask.reshape(height, width)
    return mask


def apply_mask(image: Image, mask: np.ndarray):
    # berry_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    # flower_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    # leaves_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    # assert 0.0 <= alpha <= 1.0, f'Alpha must be in [0, 1]. Got: {alpha}'

    result_mask = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

    result_mask[mask == 1] = MASK_COLORS['berry']
    result_mask[mask == 2] = MASK_COLORS['flower']
    result_mask[mask == 3] = MASK_COLORS['leaf']
    result_mask = cv2.cvtColor(result_mask, cv2.COLOR_RGB2RGBA)
    result_mask[mask == 0] = [0, 0, 0, 0]

    image = np.array(image.convert('RGBA'))
    image = cv2.addWeighted(image, 0.8, result_mask, 0.3, 0.0)

    return Image.fromarray(image).convert('RGB')


def visualize_results(
        image: Image,
        json_results,
        draw_berries=True,
        draw_flowers=True,
        draw_healthy_leaves=True,
        draw_mask=True,
        leaves_diseases=None
):
    if draw_mask:
        mask = get_mask(
            json_results['mask']['b64mask'],
            json_results['mask']['width'],
            json_results['mask']['height']
        )
        image = apply_mask(image, mask)
    draw = ImageDraw(image)
    if draw_berries:
        barries = json_results['berries']
        for bbox, ripeness in zip(barries['bboxes'], barries['ripeness']):
            x1, y1, x2, y2 = bbox
            hue = (1 - ripeness) * 0.3
            color = tuple(
                (
                    np.array(colorsys.hsv_to_rgb(hue, 1, 1)) * 255
                ).astype(np.int32)
            )
            ripeness = int(ripeness * 100)
            draw_bounding_box_on_image(
                draw, x1, y1, x2, y2, color, BBOX_WIDTH, [f'{ripeness}%']
            )

    if draw_flowers:
        flowers = json_results['flowers']
        for bbox in flowers['bboxes']:
            x1, y1, x2, y2 = bbox
            draw_bounding_box_on_image(
                draw, x1, y1, x2, y2, COLOR_FLOWER, BBOX_WIDTH
            )

    leaves = json_results['leaves']
    for bbox, src_state in zip(leaves['bboxes'], leaves['states']):
        state = MAP_LEAVES_STATE[src_state]
        x1, y1, x2, y2 = bbox
        color = COLOR_STATE_MAP[src_state]
        if state == 'Здоровый' and draw_healthy_leaves or \
                leaves_diseases is not None and state in leaves_diseases or\
                leaves_diseases is None:
            draw_bounding_box_on_image(draw, x1, y1, x2, y2, color, BBOX_WIDTH)
    return image


def compute_mean_ripeness(json_results):
    berries = json_results['berries']
    if len(berries['ripeness']) == 0:
        return None
    return np.mean(berries['ripeness'])


def sinlge_layout(url):
    st.sidebar.markdown('---')
    st.sidebar.subheader('Параметры визуализации')
    draw_flowers = st.sidebar.checkbox('Рамки цветов', value=True)
    draw_berries = st.sidebar.checkbox('Рамки ягод', value=True)
    draw_healthy_leaves = st.sidebar.checkbox(
        'Рамки здоровых листьев', value=True
    )
    draw_mask = st.sidebar.checkbox('Маска сегментации', value=True)

    image_uploaded = st.file_uploader(
        'Загрузите изображение',
        type=['jpg', 'jpeg', 'png']
    )
    if image_uploaded is not None:
        mp_encoder = MultipartEncoder(
            fields={
                'image': (
                    image_uploaded.name, image_uploaded, image_uploaded.type
                )
            }
        )
        response = requests.post(url, data=mp_encoder, headers={
            'Content-Type': mp_encoder.content_type
        })
        json_results = response.json()

        image = Image.open(image_uploaded)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Изображение')
            container = st.empty()

        with col2:
            st.subheader('Результат анализа')
            growth_state = MAP_GROWTH_PHASE[json_results['growth_phase']]
            st.write(f'🌱 **Фаза роста:** {growth_state}')

            ripeness = compute_mean_ripeness(json_results)
            if ripeness is None:
                st.write(f'🍓 **Ягод не найдено**')
            else:
                ripeness = int(ripeness * 100)
                st.write(f'🍓 **Средняя зрелость плодов:** {ripeness}%')


            diseases = json_results['leaves']['states']
            diseases_uniq = list(set(diseases).difference({'healthy'}))
            options = dict()
            if len(diseases_uniq) > 0:
                st.markdown(f'⚠️**Найденные заболевания:**')
                for dis in diseases_uniq:
                    dis = MAP_LEAVES_STATE[dis]
                    options[dis] = st.checkbox(dis, value=True)
            else:
                st.write(f'✅ **Растения выглядят здоровыми**')


        target_dis = []
        for dis, opt in options.items():
            if opt:
                target_dis.append(dis)
        with container:
            st.image(
                visualize_results(
                    image, json_results, draw_berries, draw_flowers,
                    draw_healthy_leaves, draw_mask, target_dis
                )
            )


def multiple_layout(url):
    image_loader = st.file_uploader(
        'Загрузите изображения',
        accept_multiple_files=True,
        type=['jpg', 'jpeg', 'png']
    )
    pb = st.progress(0)
    for i, image in enumerate(image_loader, start=1):
        mp_encoder = MultipartEncoder(
            fields={
                'image': (image.name, image, image.type)
            }
        )
        response = requests.post(url, data=mp_encoder, headers={
            'Content-Type': mp_encoder.content_type})
        display_results(Image.open(image), image.name, response.json())
        pb.progress(i / len(image_loader))


def display_results(image: Image, img_fname, json_results):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Изображение')
        st.write(img_fname)
        st.image(visualize_results(image, json_results))

    with col2:
        st.subheader('Результат анализа')
        growth_state = MAP_GROWTH_PHASE[json_results['growth_phase']]
        st.write(f'🌱 **Фаза роста:** {growth_state}')

        ripeness = compute_mean_ripeness(json_results)
        if ripeness is None:
            st.write(f'🍓 **Ягод не найдено**')
        else:
            ripeness = int(ripeness * 100)
            st.write(f'🍓 **Средняя зрелость плодов:** {ripeness}%')

        diseases = json_results['leaves']['states']
        diseases_uniq = list(set(diseases).difference({'healthy'}))
        if len(diseases_uniq) > 0:
            s = ''
            for dis in diseases_uniq:
                s += f'1. {MAP_LEAVES_STATE[dis]}\n'
            st.markdown(f'⚠️**Найденные заболевания:**\n\n{s[:-1]}')
        else:
            st.write(f'✅ **Растения выглядят здоровыми**')


def main(config):
    url = f"http://{config['url']}:{config['port']}/predict"
    st.set_page_config(
        page_title='Digital Strawberry',
        page_icon='icon.jpg',
        layout='wide'
    )
    st.title('Digital Strawberry')
    st.sidebar.subheader('Выберите режим работы')
    radio = st.sidebar.radio(
        '', options=['Одно изображение', 'Пакетная обработка']
    )
    if radio == 'Одно изображение':
        sinlge_layout(url)
    else:
        multiple_layout(url)


if __name__ == '__main__':
    main(read_config())
