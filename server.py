from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Tuple
from pydantic import Field, conlist
from PIL import Image
import numpy as np
import cv2
import io
import base64
from functools import lru_cache
import json

from strawberry_pipeline import CommonPipeline


import os

print(os.path.exists('models_weigts/sw_leaf_diseases.pt'))
print(os.listdir('.'))



@lru_cache()
def get_pipeline():
    return CommonPipeline(
        'models_weigts/sw_leaf_diseases.pt',
        'models_weigts/plane_matter.pt',
        'models_weigts/classifier_berry_status_224x224.pt'
    )


app = FastAPI()


class BboxesModel(BaseModel):
    bboxes: List[conlist(int, min_items=4, max_items=4)]


class BboxesBerry(BboxesModel):
    ripeness: List[float]


class BboxesFlower(BboxesModel):
    pass


class BboxesLeaf(BboxesModel):
    states: List[str]


class Response(BaseModel):
    growth_phase: str
    berries: BboxesBerry
    flowers: BboxesFlower
    leaves: BboxesLeaf



def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


def stub_predict(image: Image) -> Response:
    bboxes_berry = [[
        int(0.1 * image.size[0]),
        int(0.2 * image.size[1]),
        int(0.3 * image.size[0]),
        int(0.3 * image.size[1])
    ]]
    ripeness_berry = [0.1]

    bboxes_flower = [[
        int(0.2 * image.size[0]),
        int(0.5 * image.size[1]),
        int(0.3 * image.size[0]),
        int(0.6 * image.size[1])
    ]]

    bboxes_leaves = [
        [
            int(0.7 * image.size[0]),
            int(0.8 * image.size[1]),
            int(0.75 * image.size[0]),
            int(0.85 * image.size[1])
        ],
        [
            int(0.78 * image.size[0]),
            int(0.81 * image.size[1]),
            int(0.81 * image.size[0]),
            int(0.85 * image.size[1])
        ],
        [
            int(0.8 * image.size[0]),
            int(0.1 * image.size[1]),
            int(0.91 * image.size[0]),
            int(0.2 * image.size[1])
        ],
    ]
    states_leaves = ['healthy', 'angular_leafspot', 'leaf_scorch']

    response = Response(
        growth_phase='pre-flower',
        berries=BboxesBerry(bboxes=bboxes_berry, ripeness=ripeness_berry),
        flowers=BboxesFlower(bboxes=bboxes_flower),
        leaves=BboxesLeaf(bboxes=bboxes_leaves, states=states_leaves)
    )
    print(response)
    return response


def real_predict(image: Image) -> Response:
    pipeline = get_pipeline()
    pipeline_out = pipeline(np.array(image.convert('RGB')))
    bboxes_berry = []
    ripeness_berry = []
    for berry_rec in pipeline_out['berries']:
        bboxes_berry.append(berry_rec['box'])
        ripeness_berry.append(berry_rec['status'])

    bboxes_flower = []
    for flower_rec in pipeline_out['inflorescences']:
        bboxes_flower.append(flower_rec['box'])

    bboxes_leaves = []
    states_leaves = []
    for leaf_rec in pipeline_out['leafs']:
        bboxes_leaves.append(leaf_rec['box'])
        states_leaves.append(leaf_rec['status'].lower())

    response = Response(
        growth_phase=pipeline_out['growth_phase'],
        berries=BboxesBerry(bboxes=bboxes_berry, ripeness=ripeness_berry),
        flowers=BboxesFlower(bboxes=bboxes_flower),
        leaves=BboxesLeaf(bboxes=bboxes_leaves, states=states_leaves)
    )

    return response

    # print(pipeline_out['growth_phase'])
    # print(pipeline_out['mask_width'])
    # print(pipeline_out['mask_width'])

    # [{'box': [250, 532, 288, 572], 'status': 1.0},
    #  {'box': [281, 488, 335, 552], 'status': 1.0},
    #  {'box': [307, 217, 370, 264], 'status': 0.29629629850387573}]

    # [{'box': [1, 140, 95, 230], 'status': 'healthy'},
    #  {'box': [102, 168, 228, 265], 'status': 'healthy'},
    #  {'box': [155, 445, 228, 521], 'status': 'healthy'},
    #  {'box': [71, 38, 212, 107], 'status': 'healthy'},
    #  {'box': [58, 441, 140, 531], 'status': 'healthy'},
    #  {'box': [730, 552, 811, 660], 'status': 'healthy'},
    #  {'box': [84, 130, 177, 191], 'status': 'healthy'},
    #  {'box': [15, 231, 105, 302], 'status': 'healthy'},
    #  {'box': [547, 54, 614, 105], 'status': 'healthy'},
    #  {'box': [360, 320, 461, 385], 'status': 'healthy'},
    #  {'box': [505, 67, 611, 114], 'status': 'healthy'},
    #  {'box': [472, 124, 548, 170], 'status': 'Leaf_scorch'},
    #  {'box': [904, 124, 997, 181], 'status': 'Leaf_scorch'},
    #  {'box': [280, 347, 335, 397], 'status': 'healthy'},
    #  {'box': [680, 308, 731, 382], 'status': 'healthy'},
    #  {'box': [568, 345, 630, 400], 'status': 'healthy'},
    #  {'box': [135, 368, 185, 421], 'status': 'healthy'}]

    # [{'box': [352, 262, 751, 644]}, {'box': [334, 687, 570, 948]}]

    # berry
    # 1000
    # 1000

# {
#             'berries': berries,
#             'leafs': leafs,
#             'inflorescences': inflorescences,
#             'growth_phase': self.estimate_growth_phase(image),
#             'mask': base64.b64encode(global_mask),
#             'mask_width': global_mask.shape[1],
#             'mask_height': global_mask.shape[0]
#         }


@app.post('/predict', response_model=Response)
def get_predict(image: UploadFile = File(...)):
    # Load the image
    # img = base64str_to_PILImage(d.base64str)
    # img = d.b64image
    # img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), np.uint8))
    img = Image.open(image.file)
    response = real_predict(img)
    # img.save('lol.jpg')
    # with open('lol.jpg', 'wb') as f:
    #     f.write(image.file)
    # response = stub_predict(img)
    return response
