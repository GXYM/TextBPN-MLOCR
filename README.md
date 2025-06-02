# TextBPN-MLOCR: Advanced Multi-Lingual Scene Text Detection

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/somos99/TextBPN-MLOCR)

**Enhanced version of TextBPN++** for robust scene text detection across multiple languages and artistic fonts. Trained on large-scale synthetic and real-world text datasets for superior performance.

## Key Features ✨
- **Multi-Lingual Support**: Arabic, Bangla, Chinese, Japanese, Korean, Latin, Hindi
- **Artistic Text Detection**: Handles stylized and decorative fonts effectively
- **Optimized Performance**: Supports modern NVIDIA GPUs (H100/H800/H20)
- **Large-scale Training**: 
  - 1.5M+ synthetic text samples 
  - 500K+ real-world text samples

## Hardware Requirements
- **GPU**: NVIDIA H100, H800, H20 (CUDA 12.2 compatible)
- **Python**: ≥ 3.9
- **CUDA**: 12.2

## Model Download
Download pre-trained models from HuggingFace Hub:  
📦 [https://huggingface.co/somos99/TextBPN-MLOCR](https://huggingface.co/somos99/TextBPN-MLOCR)

## Installation

### From PyPI
```bash
pip install textbpn-mlocr
```

### From DCN with CUDA
```bash
pip install textbpn-mlocr
```

## Usage
```
import datetime
import json
import logging
import torch
from typing import List
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ocr.ocr_detection import FrameOCR


if __name__ == "__main__":
    # model
    torch.cuda.set_device(0)
    model_path = './models/TextBPN_deformable_resnet50_best2.pth'
    detect_model = FrameOCR(model_path, backbone="deformable_resnet50", use_gpu=True, need_layout=True, test_speed=False)
     
    test_img = "test.jpg"
    raw_images = cv2.imread(test_img)
    if len(raw_images.shape) == 2:
        raw_images = cv2.cvtColor(raw_images, cv2.COLOR_GRAY2BGR)

    out_puts = detect_model.detect([raw_images])
    print(out_puts)
```


## References
```
@inproceedings{zhang2021adaptive,
  title={Adaptive boundary proposal network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1305--1314},
  year={2021}
}

@article{zhang2023arbitrary,
  title={Arbitrary shape text detection via boundary transformer},
  author={Zhang, Shi-Xue and Yang, Chun and Zhu, Xiaobin and Yin, Xu-Cheng},
  journal={IEEE Transactions on Multimedia},
  volume={26},
  pages={1747--1760},
  year={2023},
  publisher={IEEE}
}
```

## License
This project is licensed under the MIT License.

## Acknowledgements

This project is based on the original TextBPN++ project.

Feel free to star & fork! For any questions, please open an issue or pull request!

