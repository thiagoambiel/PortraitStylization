import torch

import os
import argparse
import textwrap

from PIL import Image
from style_transfer import StyleTransfer


def arg_info(arg):
    defaults = StyleTransfer.stylize.__kwdefaults__
    default_types = StyleTransfer.stylize.__annotations__
    return {'default': defaults[arg], 'type': default_types[arg]}


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
        description='Stylize images given one or more style images using PortraitStylization API.',

        epilog=f'Usage:\n'
               f'  python {os.path.basename(__file__)} portrait.jpg starry_night.jpg\n'
               f'  python {os.path.basename(__file__)} portrait.jpg starry_night.jpg -cw 0.05 -fw 0.25\n'
               f'  python {os.path.basename(__file__)} landscape.jpg starry_night.jpg -cw 0.015 -fw 0.0',

        formatter_class=HelpFormatter
    )

    parser.add_argument("content", type=str,
                        help="Path to input image of human portrait.")

    parser.add_argument('styles', type=str, nargs='+',
                        help='Path to style images. (paintings are recommended)')

    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="\nDevice to run the StyleTransfer model.")

    parser.add_argument("--pooling", "-p", type=str, default='max',
                        choices=['max', 'average', 'l2'],
                        help="\nStyleTransfer VGG model pooling model.")

    parser.add_argument('--style-weights', '-sw', type=float, nargs='+', default=None,
                        help='\nThe relative weights for each style image.')

    parser.add_argument('--content-weight', '-cw', **arg_info("content_weight"),
                        help='\nThe content image weight.')

    parser.add_argument('--face-weight', '-fw', **arg_info("face_weight"),
                        help='\nThe weight of faces in content image.')

    parser.add_argument('--mesh-weight', '-mw', **arg_info("mesh_weight"),
                        help='\nThe weight of face meshes in content image. (use only for fine adjustments)')

    parser.add_argument('--tv-weight', '-tw', **arg_info("tv_weight"),
                        help='\nThe smoothing weight.')

    parser.add_argument('--min-scale', '-min_s', **arg_info('min_scale'),
                        help='\nThe minimum scale in pixels of images when in stylization. (in pixels)')

    parser.add_argument('--end-scale', '-s', **arg_info("end_scale"),
                        help='\nThe final scale of stylized image. (in pixels)')

    parser.add_argument('--iterations', '-i', **arg_info('iterations'),
                        help='\nThe number of iterations per scale.')

    parser.add_argument('--initial-iterations', '-ii', **arg_info('initial_iterations'),
                        help='\nThe number of iterations on the first scale.')

    parser.add_argument('--step-size', '-ss', **arg_info('step_size'),
                        help='\nThe step size. (learning rate)')

    parser.add_argument('--avg-decay', '-ad', **arg_info('avg_decay'),
                        help='\nThe EMA decay rate for iterate averaging.')

    parser.add_argument('--init', **arg_info('init'),
                        choices=['content', 'gray', 'uniform', 'style_mean'],
                        help='\nThe image initialization mode.')

    parser.add_argument('--style-size', **arg_info('style_size'),
                        help='\nThe fixed scale of the style at different content scales.')

    parser.add_argument('--style-scale-fac', **arg_info('style_scale_fac'),
                        help='\nThe relative scale of the style to the content.')

    parser.add_argument('--padding-scale-fac', '-ps', **arg_info("padding_scale_fac"),
                        help='\nThe padding factor to be applied to the bouncing boxes of detected faces.')

    parser.add_argument('--crop-faces', '-cf', action='store_true',
                        help='\nCrop detected faces before passing to facial models.')

    parser.add_argument('--square-faces', '-sf', action='store_true',
                        help='\nResize detected faces bouncing boxes for width == height.')

    parser.add_argument("--save-path", "-o", **arg_info("save_path"),
                        help="\nPath to save result image.")

    parser.add_argument('--save-every', **arg_info("save_every"),
                        help='Save the image every N iterations.')

    args = parser.parse_args()

    st = StyleTransfer(device=torch.device(args.device), pooling=args.pooling)

    content_image = Image.open(args.content)
    style_images = [Image.open(img) for img in args.styles]

    st.stylize(
        content_image=content_image, style_images=style_images,
        style_weights=args.style_weights, content_weight=args.content_weight,
        face_weight=args.face_weight, mesh_weight=args.mesh_weight, tv_weight=args.tv_weight,
        min_scale=args.min_scale, end_scale=args.end_scale,
        iterations=args.iterations, initial_iterations=args.initial_iterations,
        step_size=args.step_size, avg_decay=args.avg_decay, init=args.init,
        style_size=args.style_size, style_scale_fac=args.style_scale_fac, padding_scale_fac=args.padding_scale_fac,
        crop_faces=args.crop_faces, square_faces=args.square_faces,
        save_path=args.save_path, save_every=args.save_every
    )
