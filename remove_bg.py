from typing import Tuple

import os
import argparse

from PIL import Image, ImageColor
from matplotlib import colors

import torch
import numpy as np

from torch.nn import functional as F
from torchvision import transforms

from models.modnet import MODNet
from collections import OrderedDict

from utils.modules import scale_boundaries


class BackgroundRemoval:
    def __init__(self, weights_path: str, device=torch.device("cpu")):

        self.device = device

        self.model = MODNet().to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device)
        state_dict = OrderedDict({key.replace("module.", ""): value for key, value in state_dict.items()})

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    @staticmethod                # Red, Green, Blue
    def to_rgb(color: str) -> Tuple[int, int, int]:
        if "#" in color:
            return ImageColor.getcolor(color, "RGB")
        else:
            return np.array(colors.to_rgb(color)) * 255

    @staticmethod
    def gen_background(image_size, color=(0, 0, 0)) -> np.ndarray:
        background = np.zeros([*image_size], dtype=np.uint8)
        background[:, :, ] = color

        return background

    def gen_alpha(self, img: np.ndarray) -> np.ndarray:
        img = self.normalize(img).to(self.device)

        img_h, img_w = img.shape[1:3]
        img_rh, img_rw = scale_boundaries(img_h, img_w, ref_size=512)

        img = F.interpolate(img[None], size=(img_rh, img_rw), mode='area')

        matte = self.model(img)

        matte = F.interpolate(matte, size=(img_h, img_w), mode='area').squeeze()
        matte = matte.cpu().detach().numpy()

        alpha = np.stack((matte,) * 3, axis=-1).astype(float)

        return alpha

    def remove_background(self, img: Image,
                          alpha: np.ndarray = None,
                          bg_color: str = "black",
                          bg_texture: Image = None,
                          bt_fac: float = 0.5,
                          fg_color: str = None,
                          fg_fac: float = 0.2) -> Image:

        img = np.array(img)

        if alpha is None:
            alpha = self.gen_alpha(img)

        if fg_color:
            fg_color = self.to_rgb(fg_color)
            overlay = self.gen_background(img.shape, color=fg_color)
            img = (img * (1 - fg_fac)) + (overlay * fg_fac)

        bg_color = self.to_rgb(bg_color)
        background = self.gen_background(img.shape, color=bg_color)

        if bg_texture:
            bg_texture = np.array(bg_texture.resize(img.shape[0:2]))
            background = (background * (1 - bt_fac) + bg_texture * bt_fac)

        background = background

        foreground = img * alpha
        background = background * (1 - alpha)

        result = foreground + background

        return Image.fromarray(np.uint8(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Background removal tool optimized for '
                    'human portrait images with MODNet as backend.',

        epilog=f'Usage:\n'
               f'  python {os.path.basename(__file__)} -i input.jpeg\n'
               f'  python {os.path.basename(__file__)} -i input.jpeg -bc BLUE\n'
               f'  python {os.path.basename(__file__)} -i input.jpeg -bc #ff0000 -fc #0000FF\n'
               f'  python {os.path.basename(__file__)} -i input.jpeg -fc #0000FF -fc_fac 0.5',

        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--input", "-i", help="Path to input image of human portrait.", type=str, required=True)
    parser.add_argument("--output", "-o", help="Path to save result image.", type=str, required=True)

    parser.add_argument("--back-color", "-bc", help="Color to be applied to background.", type=str, default="black")
    parser.add_argument("--back-texture", "-bt", help="Path to texture image to be applied to background.", type=str)
    parser.add_argument("--back-texture-factor", "-bt-fac",
                        help="Intensity of background texture overlay.", type=float, default=0.5)

    parser.add_argument("--fore-color", "-fc", help="Color to be applied to foreground.", type=str)
    parser.add_argument("--fore-color-factor", "-fc-fac",
                        help="Intensity of foreground color overlay.", type=float, default=0.2)

    parser.add_argument("--weights", "-w", help="MODNet weights for human segmentation.", type=str, required=True)
    parser.add_argument("--device", "-d", help="Device to run the MODNet model.", type=str, default="cpu")

    args = parser.parse_args()

    background_removal = BackgroundRemoval(weights_path=args.weights, device=args.device)

    print(f"[*] Processing input image file: {args.input}")
    input_img = Image.open(args.input)

    texture = None
    if args.back_texture:
        print(f"[*] Loading texture image file: {args.back_texture}")
        texture = Image.open(args.back_texture)

    result_img = background_removal.remove_background(
        img=input_img,
        bg_color=args.back_color,
        bg_texture=texture,
        bt_fac=args.back_texture_factor,
        fg_color=args.fore_color,
        fg_fac=args.fore_color_factor
    )

    result_img.save(args.output)
    print(f"[*] Output file saved to: {args.output}")
