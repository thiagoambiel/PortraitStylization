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

First we pass the original image to the [MODNet Human Segmentation Model](#modnet-background-removal)
to generate an alpha layer and remove the background. Then the image
is forwarded through the [FaceNet](#facial-identification-loss), 
[FaceMesh](#facial-meshes-loss) and **VGG19** Models for feature extraction.
The **VGG19** Model is also used to extract the features from desired style images.
And finally, all the extracted features are joined to create the **Content**, 
**Facial** and **Style** losses.

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

## Experiments
The `content_weight` parameter focus on general image content,
while the `face_weight` parameter focus on specific facial attributes.
`mesh_weight` is helpful when adjusting finer details on result faces.

**Note**: `crop_faces` parameter is set to **False** in all the experiments below. 

<table>
  <tbody>
    <tr>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/experiments/all_models_default_params.png?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/experiments/w_o_vgg_model_default_params.png?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/experiments/w_o_facenet_model_default_params.png?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/experiments/w_o_facemesh_model_default_params.png?raw=true" width="150"/></th>
    </tr>
    <tr>
      <td>
        <ul>
          <li><nobr><b>content_weight</b>: 0.05</nobr></li>
          <li><nobr><b>face_weight</b>: 0.25</nobr></li>
          <li><nobr><b>mesh_weight</b>: 0.015</nobr></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><nobr><b>content_weight</b>: 0.0</nobr></li>
          <li><nobr><b>face_weight</b>: 0.25</nobr></li>
          <li><nobr><b>mesh_weight</b>: 0.015</nobr></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><nobr><b>content_weight</b>: 0.05</nobr></li>
          <li><nobr><b>face_weight</b>: 0.0</nobr></li>
          <li><nobr><b>mesh_weight</b>: 0.015</nobr></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><nobr><b>content_weight</b>: 0.05</nobr></li>
          <li><nobr><b>face_weight</b>: 0.25</nobr></li>
          <li><nobr><b>mesh_weight</b>: 0.0</nobr></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

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


+ `-cw 0.05` (`--content-weight`): The **Content Loss** weight. It defines how similar 
the content of result image will be when compared with the content image.


+ `-fw 0.25` (`--face-weight`): The **FaceID Loss** weight. It defines how similar
the detected faces in result image will be when compared with the content image.


You can run `python stylize.py --help` for more info.

## Example Results

**Note**: It also works with multiple faces in the same image.

<table>
  <tbody>
    <tr>
      <th>Content</th>
      <th>Style</th>
      <th>Result</th>
    </tr>
    <tr>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/original/example1.jpeg?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/styles/eletricity.jpg?raw=true" height="145"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/results/example1.png?raw=true" width="150"/></th>
    </tr>
    <tr>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/original/example2.jpeg?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/styles/abstract2.jpg?raw=true" height="145"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/results/example2.png?raw=true" width="150"/></th>
    </tr>
    <tr>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/original/example3.jpeg?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/styles/oil_painting_couple.jpeg?raw=true" height="145"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/results/example3.png?raw=true" width="150"/></th>
    </tr>
    <tr>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/original/example4.jpeg?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/styles/abstract.jpg?raw=true" height="145"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/results/example4.png?raw=true" width="150"/></th>
    </tr>
    <tr>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/original/example5.jpeg?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/styles/abstract3.jpg?raw=true" height="145"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/results/example5.png?raw=true" width="150"/></th>
    </tr>
    <tr>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/original/example6.jpeg?raw=true" width="150"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/styles/flames.jpg?raw=true" height="145"/></th>
      <th><img src="https://github.com/thiagoambiel/PortraitStylization/blob/colab/assets/examples/results/example6.png?raw=true" width="150"/></th>
    </tr>
  </tbody>
</table>

## Acknowledgements
Thanks to the authors of these amazing projects.
+ [https://github.com/crowsonkb/style-transfer-pytorch](https://github.com/crowsonkb/style-transfer-pytorch)
+ [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
+ [https://github.com/thepowerfuldeez/facemesh.pytorch](https://github.com/thepowerfuldeez/facemesh.pytorch) 
+ [https://github.com/ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)

## License
PortraitStylization is released under the MIT license. Please see the [LICENSE](https://github.com/thiagoambiel/PortraitStylization/blob/main/LICENSE) file for more information.
