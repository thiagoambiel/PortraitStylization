![](https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/portrait_stylization_banner.png?raw=true)

[![arXiv](https://img.shields.io/badge/arXiv-1508.06576-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/1508.06576)
[![Open with Colab](https://img.shields.io/badge/Open_In_Colab-0?style=for-the-badge&logo=GoogleColab&color=525252)](https://colab.research.google.com/github/thiagoambiel/PortraitStylization/blob/colab/notebooks/PortraitStylization_Demo.ipynb)

Based on the improvements of [Katherine Crowson (Neural style transfer in PyTorch)](https://github.com/crowsonkb/style-transfer-pytorch) 
on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576),
This repository brings some changes to enhance the style transfer results on images that contains human faces.

# How it Works?
![](https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/stylization_diagram.png?raw=true)
<p align="center">
  <b>Figure 1:</b> Stylization Diagram.
</p>
  
## Facial Identification Loss
Using [FaceNet Inception Resnet Model (Tim Esler)](https://github.com/timesler/facenet-pytorch), with VGGFace2 Pretrained weights, We can implement a **FaceID Loss**,
comparing the internal representations of the model from content image and result image with **MSE (Mean Squared Error)**,
like in the *VGG Model Content Loss*.

The FaceID Loss weight can be controlled through `face_weight` argument.

## Facial Meshes Loss
The **FaceMesh Loss** works like the **FaceID Loss**, but it uses only the last layer output
from [FaceMesh Model (George Grigorev)](https://github.com/thepowerfuldeez/facemesh.pytorch)
that represents the **Facial 3D Meshes**. Use only for fine adjustments on relevant expression
attributes like mouth opening.

The FaceMesh Loss weight can be controlled through `face_mesh` argument.

## MODNet Background Removal

The `BackgroundRemoval` class uses [MODNet: Trimap-Free Portrait Matting in Real Time](https://github.com/ZHKKKe/MODNet)
as backend, and uses human matting pretrained weights provided by the author. 
You can remove the background from input image for better results, but it's not necessary.

The MODNet Model can be used through the [`remove_bg.py`](#background-removal) script and the [`BackgroundRemoval`](#load-as-module) class.

## Installation

Here you can find instructions to install the project through conda.
We'll create a new environment, and install the required dependencies.

First, clone the repository locally:
```bash
git clone https://github.com/thiagoambiel/PortraitStylization.git
cd PortraitStylization
```

Create a new conda environment called `portrait_stylization`:
```bash
conda create -n portrait_stylization python=3.7
conda activate portrait_stylization
```

Now install the project dependencies:
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Basic Usage
You can use the CLI tools `remove_bg.py` and `stylize.py` or import the classes 
`BackgroundRemoval` and `StyleTransfer`. Both methods will download the 
**VGG19 Weights (548 MB)** and **FaceNet VGGFace2 Weights (107 MB)**
at the first run.

Input images will be converted to sRGB when loaded, and output images have the sRGB color space.
Alpha channels in the inputs will be ignored.

### Load as Module
The `StyleTransfer` and `BackgroundRemoval` classes can be used on an interactive
python session or in a common script.  
```python
from PIL import Image
from style_transfer import StyleTransfer
from remove_bg import BackgroundRemoval

# Load content image.
original_image = Image.open("content.jpg")

# Load MODNet and remove content image background.
background_removal = BackgroundRemoval("./weights/modnet.pth", device="cpu")

content_image = background_removal.remove_background(
    img=original_image,
    bg_color="black",
)

# Load style images.
style_images = [
    Image.open("style_1.jpg"), 
    Image.open("style_2.jpg"),
]

# Load and run style transfer module.
st = StyleTransfer(device="cpu", pooling="max")

result_image = st.stylize(
    content_image=content_image, 
    style_images=style_images,
    content_weight=0.05,
    face_weight=0.25,
)

# Save result to disk.
result_image.save("out.png")
```

### Run from Command Line

+ ### Background Removal
```bash
python remove_bg.py content.jpg output.jpg 
```

+ `content.jpg`: The input image to remove the background.


+ `output.jpg`: The path to save the result image.

You can run `python remove_bg.py --help` for more info.

+ ### Style Transfer
```bash
python stylize.py input.jpg style_1.jpg style_2.jpg -o out.png -cw 0.05 -fw 0.25
```
+ `input.jpg`: The input image to be stylized.


+ `style_N.jpg`: The style images that will be used to stylize the content
image, need to be at least one.


+ `-o out.png` (`--save-path`): The path to save the result image.


+ `-cw 0.05` (`--content-weight`): The **Content Loss** weight. Define how similar 
the content of result image will be when compared with the content image.


+ `-fw 0.25` (`--face-weight`): The **FaceID Loss** weight. Define how similar
the detected faces in result image will be when compared with the content image.


You can run `python stylize.py --help` for more info.

## Acknowledgements
Thanks to the authors of these amazing projects.
+ [https://github.com/crowsonkb/style-transfer-pytorch](https://github.com/crowsonkb/style-transfer-pytorch)
+ [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
+ [https://github.com/thepowerfuldeez/facemesh.pytorch](https://github.com/thepowerfuldeez/facemesh.pytorch) 
+ [https://github.com/ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)

## License
PortraitStylization is released under the MIT license. Please see the [LICENSE](https://github.com/thiagoambiel/PortraitStylization/blob/main/LICENSE) file for more information.
