import numpy as np
import torch
import torchvision
import base64
from io import BytesIO
from PIL import Image, ImageDraw

from yolact_inference import YOLACTModel
from preprocess_utils import create_square_crop_by_detection, TensorRotate


class CommonPipeline(object):
    colors_map = {
        0: (220, 10, 20),   # Soft red
        1: (10, 20, 200),   # Soft blue
        2: (26, 200, 5),    # Soft greed
        3: (120, 120, 135)  # Gray
    }

    matter_classes = [
        'berry',
        'first_leaves',
        'flower',
        'flowers_falling',
        'mustaches',
        'pre - berry',
        'pre - flower',
        'seed',
    ]

    diseases_classes = [
        'Leaf_scorch',
        'angular_leafspot',
        'healthy',
        'leaf_spot',
        'powdery_mildew'
    ]

    tta_transforms = [
        TensorRotate.NONE,
        TensorRotate.ROTATE_90_CLOCKWISE,
        TensorRotate.ROTATE_180,
        TensorRotate.ROTATE_90_COUNTERCLOCKWISE
    ]

    image_preprocessing = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224), interpolation=Image.BOX),
            torchvision.transforms.ToTensor()
        ]
    )

    def __init__(self,
                 leaf_classifier: str,
                 matter_classifier: str,
                 berry_status_predictor: str):
        """
        Class constructor
        Args:
            leaf_classifier: path to traced pytorch model to detect scorch
            matter_classifier:
            berry_status_predictor:
        """
        self.yolact = YOLACTModel(device='cpu')

        self.diseases_classifier = torch.jit.load(leaf_classifier)
        self.diseases_classifier.eval()

        self.matter_classifier = torch.jit.load(matter_classifier)
        self.matter_classifier.eval()

        self.berry_status_classifier = torch.jit.load(berry_status_predictor)
        self.berry_status_classifier.eval()

    def tta_inference(self, inp: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        Test series augmentation (https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d)
        Args:
            inp: square cropped tensor
            model: model to inference

        Returns:
            Averages model output
        """
        inps = [trans(inp) for trans in self.tta_transforms]

        inps = torch.stack(inps, dim=0)
        return model(inps).detach().mean(dim=0)

    def run_leaf_classifier(self, _crop: np.ndarray) -> str:
        inp_t = self.image_preprocessing(Image.fromarray(_crop))
        # out = self.diseases_classifier(inp_t).detach()[0].numpy()
        out = self.tta_inference(inp_t, self.diseases_classifier)
        return self.diseases_classes[out.argmax()]

    def estimate_growth_phase(self, image: np.ndarray) -> str:
        inp_t = self.image_preprocessing(Image.fromarray(image)).unsqueeze(0)
        out = self.matter_classifier(inp_t).detach()[0].numpy()
        return self.matter_classes[out.argmax()]

    def get_berry_status(self, img):
        inp_t = self.image_preprocessing(Image.fromarray(img))

        berry_tta = [
            TensorRotate.ROTATE_90_CLOCKWISE,
            TensorRotate.ROTATE_90_COUNTERCLOCKWISE,
            lambda x_tensor: x_tensor.flip(2)
        ]

        berry_stat = 0
        for tr in berry_tta:
            out = self.berry_status_classifier(tr(inp_t).unsqueeze(0)).detach()
            berry_stat += out[0].argmax() / (out.size(1) - 1)
        # berry_stat = out[0].argmax() / (out.size(1) - 1)
        berry_stat /= len(berry_tta)
        # print(berry_stat)
        return float(berry_stat.numpy())

    def vizualize(self, image, masks, boxes, classes):
        viz_img = Image.fromarray(image)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            target_viz_class = 2

            if classes[i] == 2:
                masked_image = np.zeros_like(image)
                masked_image[masks[i] > 0] = image[masks[i] > 0]
                crop = create_square_crop_by_detection(masked_image, [x1, y1, x2, y2])
                leaf_status = self.run_leaf_classifier(crop)
                print(leaf_status)

                if leaf_status != 'healthy':
                    target_viz_class = 0
            elif classes[i] == 1:
                target_viz_class = 1
            elif classes[i] == 0:
                target_viz_class = 3

            draw = ImageDraw.Draw(viz_img)

            # Draw a regular rectangle
            # draw.rectangle((x1, y1, x2, y2), fill=(26, 200, 5))
            # Draw a rounded rectangle

            if classes[i] == 0:
                crop = create_square_crop_by_detection(image, [x1, y1, x2, y2])
                berry_stat = self.get_berry_status(crop)
                tc =(int(berry_stat * 255), int((1 - berry_stat) * 255), 0)

                draw.rounded_rectangle((x1, y1, x2, y2), fill=None,
                                       outline=tc,
                                       width=7, radius=15)

                # draw.rounded_rectangle((x1, y1, x2, y2), fill=None,
                #                        outline=self.colors_map[target_viz_class],
                #                        width=3, radius=15)
            else:
                draw.rounded_rectangle((x1, y1, x2, y2), fill=None,
                                       outline=self.colors_map[
                                           target_viz_class],
                                       width=9, radius=15)

        return np.array(viz_img), self.estimate_growth_phase(image)

    def get_to_viz(self, image):
        return self.vizualize(image, *self.yolact(image))

    def __call__(self, image: np.ndarray):
        """
        Calling method
        Args:
            image: RGB image in HWC format

        Returns:

        """
        masks, boxes, classes = self.yolact(image)

        berries = []
        leafs = []
        inflorescences = []
        global_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            global_mask[masks[i] > 0] = classes[i] + 1

            if classes[i] == 2:
                masked_image = np.zeros_like(image)
                masked_image[masks[i] > 0] = image[masks[i] > 0]
                crop = create_square_crop_by_detection(masked_image, [x1, y1, x2, y2])
                leaf_status = self.run_leaf_classifier(crop)
                leafs.append(
                    {
                        'box': [x1, y1, x2, y2],
                        'status': leaf_status
                    }
                )
            elif classes[i] == 0:
                crop = create_square_crop_by_detection(image, [x1, y1, x2, y2])
                berry_st = self.get_berry_status(crop)
                berries.append(
                    {
                        'box': [x1, y1, x2, y2],
                        'status': berry_st
                    }
                )
            else:
                inflorescences.append(
                    {
                        'box': [x1, y1, x2, y2]
                    }
                )

        return {
            'berries': berries,
            'leafs': leafs,
            'inflorescences': inflorescences,
            'growth_phase': self.estimate_growth_phase(image),
            'mask': base64.b64encode(global_mask),
            'mask_width': global_mask.shape[1],
            'mask_height': global_mask.shape[0]
        }


if __name__ == '__main__':
    img_path = '/home/alexey/programming/squanch_work/strawberry_project/datasets/strawberry/37786E52-F57E-4777-A9E8-D94CBFE89EAD_1_105_c.jpeg'
    img_path = '/home/alexey/programming/squanch_work/strawberry_project/datasets/strawberry/5F93DE5A-A6B3-4186-A6DF-A77E8E3EEEA8_1_105_c.jpeg'
    img = np.array(Image.open(img_path))
    pipeline = CommonPipeline(
        'models_weigts/sw_leaf_diseases.pt',
        'models_weigts/plane_matter.pt',
        'models_weigts/classifier_berry_status_224x224.pt'
    )

    res, matter = pipeline.get_to_viz(img)

    pipeline_out = pipeline(img)
    mask = np.frombuffer(base64.decodebytes(pipeline_out['mask']), dtype=np.uint8)
    mask = mask.reshape((pipeline_out['mask_height'], pipeline_out['mask_width']))
    print(mask.shape)

    Image.fromarray(res).save('res.jpg')
    Image.fromarray(mask).save('mask.png')
    print(matter)
