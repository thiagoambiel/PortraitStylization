from typing import List, Tuple
import copy

import numpy as np
from PIL import Image

import torch
from torch import nn, Tensor

from models import FaceMesh
from facenet_pytorch import InceptionResnetV1, MTCNN

from torchvision.transforms import functional as TF


class FacialFeatures(nn.Module):
    def __init__(self, layers: List[int], device=torch.device("cpu")):
        super().__init__()

        self.device = device
        self.layers = layers

        # Notable layers represents the layers where changes in the number of channels occur.
        self.notable_layers = [1, 3, 4, 5, 7, 9]

        # Load models for facial representation.
        self.facenet = self.load_facenet()
        self.facemesh = self.load_facemesh()

        # Load MTCNN Model for face detection and crop.
        self.mtcnn = MTCNN(keep_all=True, device=torch.device("cpu"))
        self.min_face_size_percent = 0.01
        self.faces_boxes = None

        self.padding_scale_fac = 0.2
        self.square_faces = True

    def load_facemesh(self) -> nn.Module:
        """Load FaceMesh model."""
        model = FaceMesh().to(self.device)
        model.load_weights("./weights/facemesh.pth")

        model.eval()
        model.requires_grad_(False)

        return model

    def load_facenet(self) -> nn.Module:
        """Load FaceNet model and change first conv layer padding to 'replicate'."""
        model = InceptionResnetV1(pretrained='vggface2').to(self.device)
        model = nn.Sequential(*list(model.children())[:14])
        model[0].conv = self._change_padding_mode(model[0].conv, 'replicate').to(self.device)

        model.eval()
        model.requires_grad_(False)

        return model

    def extract_boxes(self, image: Image):
        """Extract faces bounding boxes from input image."""
        if image:
            self.faces_boxes, _ = self.mtcnn.detect(image)

            if self.faces_boxes is not None:
                self.faces_boxes = np.int32(self.faces_boxes)

        else:
            self.faces_boxes = None

    def extract_features(self, input: Tensor, layers: List[int]) -> dict:
        """Extract features from input image with desired models layers. Can be FaceNet and FaceMesh."""
        layers = copy.deepcopy(layers)
        feats = {}

        if 14 in layers:
            _input = self.preprocess(input, model="facemesh")
            feats[14], _ = self.facemesh(_input)
            layers.remove(14)

        if any(x in layers for x in range(14)):
            _input = self.preprocess(input, model="facenet")
            for layer in range(max(layers) + 1):
                _input = self.facenet[layer](_input)

                if layer in layers:
                    feats[layer] = _input

        return feats

    @staticmethod
    def _change_padding_mode(conv: nn.Conv2d, padding_mode: str) -> nn.Module:
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)

        return new_conv

    @staticmethod
    def crop_tensor(input: Tensor, box: np.ndarray) -> Tensor:
        """Crop tensor given a bounding box."""
        return input[:, :, box[1]:box[3], box[0]:box[2]]

    @staticmethod
    def process_box(box: np.ndarray, padding_scale_fac: float, square: bool) -> Tuple[int, np.ndarray]:
        face_w, face_h = box[2] - box[0], box[3] - box[1]
        face_area = face_w * face_h

        delta = int(abs(face_w - face_h) / 2)
        padding = max(face_w, face_h) * padding_scale_fac

        padding_w = padding + (delta if face_w < face_h and square else 0)
        padding_h = padding + (delta if face_h < face_w and square else 0)

        pad_box = np.array([
            box[0] - padding_w,
            box[1] - padding_h,
            box[2] + padding_w,
            box[3] + padding_h
        ], dtype=np.int32)
        pad_box = np.clip(pad_box, 0, None)

        return face_area, pad_box

    @staticmethod
    def _get_min_size(layers: List[int]) -> int:
        last_layer = max(layer for layer in layers if layer != 14)
        min_size = 1

        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2

        return min_size

    def preprocess(self, input: Tensor, model: str) -> Tensor:
        """Preprocess input image according to the needs of each model."""
        if model == "facenet":
            if self.faces_boxes is not None:
                input = TF.resize(input, [160, 160])

            return (input.mul(255) - 127.5) / 128.0

        elif model == "facemesh":
            input = TF.resize(input, [192, 192])
            return input.mul(255) / 127.5 - 1

        else:
            raise ValueError("model must be 'facenet' or 'facemesh'")

    def forward(self, input: Tensor, layers: List[int] = None) -> dict:
        """Extract facial features from input image."""
        layers = self.layers if layers is None else layers
        input = input.to(self.device)

        if self.faces_boxes is not None:
            input_area = input.size(2) * input.size(3)

            faces_feats = []
            for box in self.faces_boxes:
                face_area, pad_box = self.process_box(box, self.padding_scale_fac, self.square_faces)

                if face_area / input_area > self.min_face_size_percent:
                    face = self.crop_tensor(input, pad_box)
                    feats = self.extract_features(face, layers=layers)

                    faces_feats.append(feats)

            feats = {}
            for layer in faces_feats[0]:
                data = torch.tensor([]).to(self.device)

                for face_feats in faces_feats:
                    data = torch.cat([data, face_feats[layer]], dim=0)

                feats[layer] = data

        else:
            feats = self.extract_features(input, layers=layers)

        return feats
