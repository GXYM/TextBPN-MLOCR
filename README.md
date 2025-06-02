# TextBPN-MLOCR: Advanced Multi-Lingual Scene Text Detection
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/somos99/TextBPN-MLOCR)
[![PyPI Version](https://img.shields.io/pypi/v/textbpn-mlocr)](https://pypi.org/project/textbpn-mlocr/)

Enhanced version of TextBPN++ for robust scene text detection across multiple languages and artistic fonts. Trained on large-scale synthetic and real-world text datasets for superior performance in diverse scenarios.

## ✨ Key Features
- **Multi-Lingual Support**: Detect text in Arabic, Bangla, Chinese, Japanese, Korean, Latin, Hindi
- **Artistic Text Handling**: Accurately processes stylized and decorative fonts
- **Optimized Performance**: Fully supports modern NVIDIA GPUs (H100/H800/H20)
- **Large-scale Training**:
  - 🧪 1.5M+ synthetic text samples
  - 📸 500K+ real-world text samples

## 🛠️ Hardware Requirements
| Component      | Requirement                         |
|----------------|-------------------------------------|
| **GPU**        | NVIDIA H100, H800, H20              |
| **CUDA**       | 12.2                                |
| **Python**     | ≥ 3.9                               |
| **OS**         | Linux (recommended)                |

## 🔽 Model Download
Download pre-trained models from HuggingFace Hub:  
[https://huggingface.co/somos99/TextBPN-MLOCR](https://huggingface.co/somos99/TextBPN-MLOCR)

## 📦 Installation
Install via PyPI:
```bash
pip install textbpn-mlocr
```

From DCN with CUDA
```bash
sh make.sh
```

## 🚀 Quick Start
```python
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


## 📖 References
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

## ⚖️ License
This project is licensed under the MIT License.

## 🙏 Acknowledgements
This project extends the original work from:
* TextBPN++: GitHub Repository
* Contributors to the TextBPN project
## Contribute & Support​​
🌟 Star us on GitHub → https://github.com/somos99/TextBPN-MLOCR  
🐛 Report issues → https://github.com/somos99/TextBPN-MLOCR/issues  
📥 Pull requests welcome!
