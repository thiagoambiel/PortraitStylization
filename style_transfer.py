from typing import List, Tuple
import time

from IPython.display import display, Pretty

import torch
from torch import Tensor
from torch.optim import Adam

from torchvision.transforms import functional as TF

import numpy as np
from PIL import Image

from models import VGGFeatures, FacialFeatures

from utils.modules import Scale, LayerApply, EMA
from utils.modules import MergeModels, Model, Layers
from utils.modules import gen_scales, size_to_fit, interpolate
from utils.modules import scale_adam, scale_boundaries, get_model_min_size

from utils.losses import TVLoss, ContentLoss, StyleLoss, SumLoss

import warnings

warnings.filterwarnings("ignore")


class StyleTransfer:
    def __init__(self, device=torch.device("cpu"), pooling: str = 'max'):
        self.device = device

        self.image = None
        self.average = None

        # The default content and style layers follow Gatys et al. (2015).
        self.content_layers = Layers(tag="vgg", layers=[22])
        self.style_layers = Layers(tag="vgg", layers=[1, 6, 11, 20, 29])

        # The facial layers were empirically selected through debugging of
        # the FaceNet Model (shallow layers are better in generalization).
        # Layer 14 represents the FaceMesh Model Landmarks.
        self.face_layers = Layers(tag="facenet", layers=[0, 1, 3, 4, 5], hidden_layers=[14])

        # The weighting of the style layers differs from Gatys et al. (2015) and Johnson et al.
        style_weights = [256, 64, 16, 4, 1]
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]

        self.vgg = VGGFeatures(
            layers=self.style_layers.layers + self.content_layers.layers,
            pooling=pooling,
            device=self.device
        )

        self.facenet = FacialFeatures(
            layers=self.face_layers.layers,
            device=self.device
        )

        self.model = MergeModels([
            Model(model=self.vgg, tag="vgg"),
            Model(model=self.facenet, tag="facenet")
        ])

        self.min_input_size = max(
            get_model_min_size(
                layers=self.content_layers.layers + self.style_layers.layers,
                notable_layers=self.vgg.notable_layers
            ),
            get_model_min_size(
                layers=self.face_layers.layers,
                notable_layers=self.facenet.notable_layers
            )
        )

    def get_image_tensor(self):
        return self.average.get().detach()[0].clamp(0, 1)

    def get_image(self, image_type='pil'):
        if self.average is not None:
            image = self.get_image_tensor()

            if image_type.lower() == 'pil':
                return TF.to_pil_image(image)

            elif image_type.lower() == 'np_uint16':
                arr = image.cpu().movedim(0, 2).numpy()
                return np.uint16(np.round(arr * 65535))

            else:
                raise ValueError("image_type must be 'pil' or 'np_uint16'")

    def initialize_image(self, content_image: Image,
                         style_images: List,
                         style_weights: List[int],
                         size: Tuple[int, int],
                         init: str = 'content',
                         device=torch.device("cpu")):

        if init == 'content':
            self.image = TF.to_tensor(content_image.resize(size, Image.LANCZOS))[None].to(device)

        elif init == 'gray':
            self.image = torch.rand([1, 3, *size]) / 255 + 0.5
            self.image.to(device)

        elif init == 'uniform':
            self.image = torch.rand([1, 3, *size]).to(device)

        elif init == 'style_mean':
            means = []
            for i, image in enumerate(style_images):
                means.append(TF.to_tensor(image).mean(dim=(1, 2)) * style_weights[i])

            self.image = torch.rand([1, 3, *size]) / 255 + sum(means)[None, :, None, None]
            self.image.to(device)

        else:
            raise ValueError("init must be one of 'content', 'gray', 'uniform', 'style_mean'")

    @staticmethod
    def calculate_losses(features: dict, layers: List[str], weights: List[float]):
        losses = []

        for layer, weight in zip(layers, weights):
            target = features[layer]
            losses.append(Scale(LayerApply(ContentLoss(target), layer), weight))

        return losses

    def process_layers(self, content_weight: float,
                       face_weight: float,
                       mesh_weight: float) -> List[Layers]:

        layers = []

        if content_weight > 0:
            layers.append(self.content_layers)

        if face_weight > 0 or mesh_weight > 0:
            face_layers = Layers(self.face_layers.tag, [])

            if face_weight > 0:
                face_layers.load(self.face_layers.layers)

            if mesh_weight > 0:
                face_layers.append(14)

            layers.append(face_layers)

        return layers

    def process_content(self, content: Tensor,
                        content_weights: List[float],
                        face_weights: List[float],
                        mesh_weight: float) -> Tuple[List[Scale], List[Layers]]:

        layers = self.process_layers(
            content_weight=sum(content_weights),
            face_weight=sum(face_weights),
            mesh_weight=mesh_weight
        )
        content_feats = self.model(content, layers)

        losses = []

        if sum(content_weights) > 0:
            content_losses = self.calculate_losses(
                features=content_feats,
                layers=list(self.content_layers),
                weights=content_weights
            )
            losses += content_losses

        if sum(face_weights) > 0:
            facial_losses = self.calculate_losses(
                features=content_feats,
                layers=list(self.face_layers),
                weights=face_weights
            )
            losses += facial_losses

        if mesh_weight > 0:
            mesh_layer = f'{self.face_layers.tag}_14'
            mesh_loss = Scale(LayerApply(ContentLoss(content_feats[mesh_layer]), mesh_layer), mesh_weight)

            losses.append(mesh_loss)

        return losses, layers

    def process_styles(self, style_images: List,
                       style_size: int,
                       scale: int,
                       style_scale_fac: float,
                       style_weights: List[float]) -> List[Scale]:

        style_targets, style_losses = {}, []

        for i, image in enumerate(style_images):
            if style_size is None:
                sw, sh = size_to_fit(image.size, round(scale * style_scale_fac))
            else:
                sw, sh = size_to_fit(image.size, style_size)

            style = TF.to_tensor(image.resize((sw, sh), Image.LANCZOS))[None]
            style = style.to(self.device)

            # Take the weighted average of multiple style targets (Gram matrices).
            style_feats = self.model(style, [self.style_layers])
            for layer in list(self.style_layers):
                target = StyleLoss.get_target(style_feats[layer]) * style_weights[i]

                if layer not in style_targets:
                    style_targets[layer] = target
                else:
                    style_targets[layer] += target

        for layer, weight in zip(list(self.style_layers), self.style_weights):
            target = style_targets[layer]
            style_losses.append(Scale(LayerApply(StyleLoss(target), layer), weight))

        return style_losses

    def stylize(self, content_image: Image.Image, style_images: List[Image.Image], *,

                style_weights: list = None,
                content_weight: float = 0.015,
                face_weight: float = 0.015,
                mesh_weight: float = 0,
                tv_weight: float = 2.,

                min_scale: int = 128,
                end_scale: int = 512,

                iterations: int = 500,
                initial_iterations: int = 1000,

                step_size: float = 0.02,
                avg_decay: float = 0.99,
                init: str = 'content',

                style_size: int = None,
                style_scale_fac: float = 1.,
                padding_scale_fac: float = 0.2,

                crop_faces: bool = False,
                square_faces: bool = False,

                plot_progress: bool = False,
                plot_every: int = 100,

                save_path: str = "./out.png",
                save_every: int = 10):

        if square_faces and not crop_faces:
            raise ValueError("To use 'square_faces', 'crop_faces' need to be True.")

        if plot_every == 0:
            raise ValueError("'plot_every' can't be zero to avoid ZeroDivisionError.")

        style_sizes = [style.size for style in style_images]
        if min([size for sizes in style_sizes for size in sizes]) < self.min_input_size:
            raise ValueError(f'Style images need to be at least {self.min_input_size}x{self.min_input_size}.')

        content_image = content_image.convert("RGB")
        style_images = [style_image.convert("RGB") for style_image in style_images]

        min_scale = min(min_scale, end_scale)

        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)
        face_weights = [face_weight / len(self.face_layers)] * len(self.face_layers)

        if style_weights is None:
            style_weights = [1 / len(style_images)] * len(style_images)
        else:
            weight_sum = sum(abs(w) for w in style_weights)
            style_weights = [weight / weight_sum for weight in style_weights]

        if len(style_images) != len(style_weights):
            raise ValueError('style_images and style_weights must have the same length')

        tv_loss = Scale(LayerApply(TVLoss(), 'input'), tv_weight)

        scales = gen_scales(min_scale, end_scale)

        cw, ch = size_to_fit(content_image.size, scales[0], scale_up=True)

        self.initialize_image(
            content_image=content_image,
            style_images=style_images,
            style_weights=style_weights,
            size=(cw, ch),
            init=init,
            device=self.device
        )

        self.model.get_model("facenet").padding_scale = padding_scale_fac
        self.model.get_model("facenet").square_faces = square_faces

        plot_size = scale_boundaries(*content_image.size, ref_size=256)
        image_display = display(content_image.resize(plot_size), display_id=True) if plot_progress else None
        status_display = display((), display_id=True)

        optimizer = None

        # Stylize the image at successively finer scales, each greater by a factor of sqrt(2).
        # This differs from the scheme given in Gatys et al. (2016).
        for scale in scales:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            cw, ch = size_to_fit(content_image.size, scale, scale_up=True)
            scaled_content = content_image.resize((cw, ch), Image.LANCZOS)

            content = TF.to_tensor(scaled_content)[None]
            content = content.to(self.device)

            self.model.get_model("facenet").extract_boxes(scaled_content if crop_faces else None)

            self.image = interpolate(self.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            self.average = EMA(self.image, avg_decay)
            self.image.requires_grad_()

            content_losses, content_layers = self.process_content(
                content=content,
                content_weights=content_weights,
                face_weights=face_weights,
                mesh_weight=mesh_weight,
            )

            style_losses = self.process_styles(
                style_images=style_images,
                style_size=style_size,
                scale=scale,
                style_scale_fac=style_scale_fac,
                style_weights=style_weights,
            )

            criterion = SumLoss([*content_losses, *style_losses, tv_loss])

            scale_optimizer = Adam([self.image], lr=step_size)

            # Warm-start the Adam optimizer if this is not the first scale.
            if scale != scales[0]:
                optimizer_state = scale_adam(optimizer.state_dict(), (ch, cw))
                scale_optimizer.load_state_dict(optimizer_state)
            optimizer = scale_optimizer

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            max_iterations = initial_iterations if scale == scales[0] else iterations
            for i in range(1, max_iterations + 1):
                start_ts = time.time()

                feats = self.model(self.image, [*content_layers, self.style_layers])
                loss = criterion(feats)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Enforce box constraints.
                with torch.no_grad():
                    self.image.clamp_(0, 1)
                self.average.update(self.image)

                end_ts = time.time()

                status = f'Size: {cw}x{ch}, iteration: {i}/{max_iterations}, '\
                         f'loss:{loss:g}, elapsed (ms): {(end_ts - start_ts) * 1000:.2f}'
                status_display.update(Pretty(status)) if status_display else print(status)

                if i % save_every == 0 and save_path:
                    result = self.get_image()
                    result.save(save_path)

                if i % plot_every == 0 and image_display:
                    result = self.get_image()
                    image_display.update(result.resize(plot_size))

            # Initialize each new scale with the previous scale's averaged iterate.
            with torch.no_grad():
                self.image.copy_(self.average.get())

        return self.get_image()
