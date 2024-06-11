<div align="center">
    <img src="assets/banner.png" width="75%" alt="Torqueo logo" align="center" />
</div>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.7, 3.8, 3.9, 3.10-efefef">
    </a>
    <a href="https://github.com/serre-lab/Warped/actions/workflows/lint.yml/badge.svg">
        <img alt="PyLint" src="https://github.com/serre-lab/Warped/actions/workflows/lint.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Warped/actions/workflows/tox.yml/badge.svg">
        <img alt="Tox" src="https://github.com/serre-lab/Warped/actions/workflows/tox.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Warped/actions/workflows/publish.yml/badge.svg">
        <img alt="Pypi" src="https://github.com/serre-lab/Warped/actions/workflows/publish.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/torqueo">
        <img alt="Pepy" src="https://static.pepy.tech/badge/torqueo">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>

**Torqueo** is a simple and hackable library for experimentation with image warping in **PyTorch**. It is designed to facilitate easy manipulation and transformation of images using various warping techniques.


# 🚀 Getting Started with Torqueo

**Torqueo** requires **Python 3.7** or newer and several dependencies, including **Numpy**. Installation is straightforward with **Pypi**:

```bash
pip install torqueo
```

With **Torqueo** installed, you can dive into image warping. The API is designed to be **intuitive**, requiring only a few hyperparameters to get started.

Example usage:

```python
import torch
import timm
from torqueo import Fisheye

transformed_images = Fisheye()(images)
```

**Starter Notebook:** <a href="https://colab.research.google.com/drive/1X_DuMAWEwE1GRMc7kyTIIg4xiiQWVJo3?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab" style="vertical-align: middle;"></a>

# Examples of transformations

Below are some examples of image transformations using **Torqueo**.

|  |  |  |
|-|-|-|
| <img src="./assets/rabbit.jpg" width="99%" alt="Original Image"><br> **Original Image** | <img src="./assets/transformations/barrel.jpg" width="99%" alt="Barrel"><br> **Barrel** | <img src="./assets/transformations/fisheye.jpg" width="99%" alt="Fisheye"><br> **Fisheye**|
| <img src="./assets/transformations/perspective.jpg" width="99%" alt="Perspective"><br> **Perspective** | <img src="./assets/transformations/pinch.jpg" width="99%" alt="Pinch"><br> **Pinch** | <img src="./assets/transformations/spherize.jpg" width="99%" alt="Spherize"><br> **Spherize**|
| <img src="./assets/transformations/stretch.jpg" width="99%" alt="Stretch"><br> **Stretch** | <img src="./assets/transformations/swirl.jpg" width="99%" alt="Swirl"><br> **Swirl** | <img src="./assets/transformations/twirl.jpg" width="99%" alt="Twirl"><br> **Twirl**|
| <img src="./assets/transformations/wave.jpg" width="99%" alt="Wave"><br> **Wave** | | |

# Authors of the code

- [**Vipul Sharma**]() - vipul_sharma@brown.edu, Brown University
- [**Thomas Fel**](https://thomasfel.fr) - thomas_fel@brown.edu, PhD Student DEEL (ANITI), Brown University
