import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ocr.ocr_detection import FrameOCR

# 初始化模型（只加载一次）
torch.cuda.set_device(0)
model_path = './models/TextBPN_deformable_resnet50_best2.pth'
detect_model = FrameOCR(model_path, backbone="deformable_resnet50", use_gpu=True, need_layout=True, test_speed=False)

def detect_and_visualize(image: Image.Image, dis_threshold: float, cls_threshold: float):
    if image is None:
        return None, None
    # 转为OpenCV格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    # 检测
    outputs = detect_model.detect([img_cv], dis_threshold=dis_threshold, cls_threshold=cls_threshold)
    res = outputs[0]
    # 可视化
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    # 画boundingBox
    if "boundingBox" in res:
        for box in res["boundingBox"]:
            draw.polygon([tuple(point) for point in box], outline="yellow", width=3)
    # 画检测到的文本框
    for line in res["lines"]:
        draw.polygon([tuple(point) for point in line["contour"]], outline="red", width=3)
        if "lineBox" in line:
            draw.polygon([tuple(point) for point in line["lineBox"]], outline="green", width=3)
    # 返回可视化图片和原始检测结果
    return vis_img, res

def clear_all():
    return None, None, None

with gr.Blocks() as demo:
    gr.Markdown("# TextBPN OCR 检测可视化")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传图片")
            dis_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="dis_threshold")
            cls_slider = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="cls_threshold")
            with gr.Row():
                btn = gr.Button("检测")
                clear_btn = gr.Button("清除")
        with gr.Column():
            output_image = gr.Image(type="pil", label="检测结果可视化")
            output_json = gr.JSON(label="检测原始结果")
    btn.click(
        fn=detect_and_visualize, 
        inputs=[input_image, dis_slider, cls_slider], 
        outputs=[output_image, output_json]
    )
    clear_btn.click(
        fn=clear_all, 
        inputs=[], 
        outputs=[input_image, output_image, output_json]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)
