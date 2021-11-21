from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from pydantic import conlist
from PIL import Image
import numpy as np
import io
import base64
from functools import lru_cache

from strawberry_pipeline import CommonPipeline


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


class Mask(BaseModel):
    b64mask: bytes
    width: int
    height: int


class Response(BaseModel):
    growth_phase: str
    berries: BboxesBerry
    flowers: BboxesFlower
    leaves: BboxesLeaf
    mask: Mask


def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


# def stub_predict(image: Image) -> Response:
#     bboxes_berry = [[
#         int(0.1 * image.size[0]),
#         int(0.2 * image.size[1]),
#         int(0.3 * image.size[0]),
#         int(0.3 * image.size[1])
#     ]]
#     ripeness_berry = [0.1]
#
#     bboxes_flower = [[
#         int(0.2 * image.size[0]),
#         int(0.5 * image.size[1]),
#         int(0.3 * image.size[0]),
#         int(0.6 * image.size[1])
#     ]]
#
#     bboxes_leaves = [
#         [
#             int(0.7 * image.size[0]),
#             int(0.8 * image.size[1]),
#             int(0.75 * image.size[0]),
#             int(0.85 * image.size[1])
#         ],
#         [
#             int(0.78 * image.size[0]),
#             int(0.81 * image.size[1]),
#             int(0.81 * image.size[0]),
#             int(0.85 * image.size[1])
#         ],
#         [
#             int(0.8 * image.size[0]),
#             int(0.1 * image.size[1]),
#             int(0.91 * image.size[0]),
#             int(0.2 * image.size[1])
#         ],
#     ]
#     states_leaves = ['healthy', 'angular_leafspot', 'leaf_scorch']
#
#     response = Response(
#         growth_phase='pre-flower',
#         berries=BboxesBerry(bboxes=bboxes_berry, ripeness=ripeness_berry),
#         flowers=BboxesFlower(bboxes=bboxes_flower),
#         leaves=BboxesLeaf(bboxes=bboxes_leaves, states=states_leaves),
#
#     )
#     print(response)
#     return response


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

    mask = Mask(
        b64mask=pipeline_out['mask'],
        width=pipeline_out['mask_width'],
        height=pipeline_out['mask_height']
    )

    response = Response(
        growth_phase=pipeline_out['growth_phase'],
        berries=BboxesBerry(bboxes=bboxes_berry, ripeness=ripeness_berry),
        flowers=BboxesFlower(bboxes=bboxes_flower),
        leaves=BboxesLeaf(bboxes=bboxes_leaves, states=states_leaves),
        mask=mask
    )

    return response


@app.post('/predict', response_model=Response)
def get_predict(image: UploadFile = File(...)):
    img = Image.open(image.file)
    response = real_predict(img)
    return response
