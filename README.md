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
