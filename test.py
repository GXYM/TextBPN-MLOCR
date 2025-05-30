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

    




