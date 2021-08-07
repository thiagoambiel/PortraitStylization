from typing import Tuple

import os
import argparse
import textwrap

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
            bg_texture = np.array(bg_texture.resize(img.shape[0:2][::-1]))
            background = (background * (1 - bt_fac) + bg_texture * bt_fac)

        foreground = img * alpha
        background = background * (1 - alpha)

        result = foreground + background

        return Image.fromarray(np.uint8(result))


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawTextHelpFormatter,
                    argparse.HelpFormatter):

    def split_lines(self, text, width):
        text = self._whitespace_matcher.sub(' ', text).strip()
        return textwrap.wrap(text, width)

    def _split_lines(self, text, width):
        lines = self.split_lines(text, width)

        if text.startswith('\n') or "help message" in text:
            lines += ['']

        return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Background removal tool optimized for '
                    'human portrait images with MODNet as backend.',

        epilog=f'Usage:\n'
               f'  python {os.path.basename(__file__)} input.png output.png\n'
               f'  python {os.path.basename(__file__)} input.png output.png -bc BLUE\n'
               f'  python {os.path.basename(__file__)} input.png output.png -bc #ff0000 -fc #0000FF\n'
               f'  python {os.path.basename(__file__)} input.png output.png -fc #0000FF -fc_fac 0.5',

        formatter_class=HelpFormatter
    )

    parser.add_argument("input", type=str,
                        help="Path to input image of human portrait.")

    parser.add_argument("output", type=str,
                        help="Path to save result image.")

    parser.add_argument("--back-color", "-bc", type=str, default="black",
                        help="\nColor to be applied to background.")

    parser.add_argument("--back-texture", "-bt", type=str,
                        help="\nPath to texture image to be applied to background.")

    parser.add_argument("--back-texture-factor", "-bt-fac",
                        help="\nIntensity of background texture overlay.", type=float, default=0.5)

    parser.add_argument("--fore-color", "-fc", type=str,
                        help="\nColor to be applied to foreground.")

    parser.add_argument("--fore-color-factor", "-fc-fac",
                        help="\nIntensity of foreground color overlay.", type=float, default=0.2)

    parser.add_argument("--weights", "-w", type=str, default="./weights/modnet.pth",
                        help="\nMODNet weights for human segmentation.")

    parser.add_argument("--device", "-d", type=str, default="cpu",
                        help="Device to run the MODNet model.")

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
