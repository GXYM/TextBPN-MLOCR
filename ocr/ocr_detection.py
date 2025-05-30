import cv2
import numpy as np
from shapely.geometry import *
import torch
from ocr.network.textnet import TextNet
from ocr.cfglib.config import config as cfg, update_config, print_config
from ocr.cfglib.option import BaseOptions
from ocr.util.augmentation import BaseTransform
from ocr.util.misc import to_device,rescale_result
from ocr.util.lib import cluster_polygons, minimum_bounding_rectangle
from ocr.util import canvas as cav
import random

class FrameOCR:
    def __init__(self, model_path, backbone="deformable_resnet50", use_gpu=True, need_layout=False, test_speed=True):
        # parse arguments
        print_config(cfg)
        self.need_layout=need_layout
        self.test_speed=test_speed

        if use_gpu:
            # 设置设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Model
        model = TextNet(is_training=False, backbone=backbone)
        model.load_model(model_path)
        # load_model(model, model_path)
        model = model.to(self.device)  # copy to cuda
        self.model= model.eval()

        self.transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        print("Init model done !!!")
    
    def detect(self, img_list):
        out_puts = []
        with torch.no_grad():
            for image in img_list:
                input_dict = dict()
                H, W, _ = image.shape
                # print(image.shape)
                img, _ = self.transform(image)

                image_tensor = torch.from_numpy(img).float()  # 将 numpy 数组转换为 PyTorch 张量
                image_tensor = image_tensor.permute(2, 0, 1)  # 将通道维度移到前面 (HWC -> CHW)
                image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度 (C, H, W -> 1, C, H, W)
                input_dict['img'] = to_device(image_tensor)

                # get detection result
                output_dict = self.model(input_dict, test_speed=self.test_speed)
                
                # visualization
                img_show = input_dict['img'][0].permute(1, 2, 0).cpu().numpy()
                img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
                contours = output_dict["py_preds"][-1].int().cpu().numpy()
                # # print(img_show.shape)
                
                # cls_preds = output_dict["fy_preds"][0].data.cpu().numpy()
                # cv2.drawContours(img_show, contours, -1, (0, 0, 255), 2)
                # cls_pred = cav.heatmap(np.array(cls_preds[0] * 255, dtype=np.uint8))
                # dis_pred = cav.heatmap(np.array(cls_preds[1] * 255, dtype=np.uint8))
                # heat_map = np.concatenate([cls_pred*255, dis_pred*255], axis=1)
                # cv2.imwrite(f"./vis/{random.randint(0, 1000)}.jpg", heat_map)

                img_show, contours = rescale_result(img_show, contours, H, W)
                
                OCRArea = 0
                lines = []
                pred_bbox = []
                for j, cont in enumerate(contours):
                    rect = cv2.minAreaRect(cont)
                    if min(rect[1][0], rect[1][1]) <= 3:
                        continue
                    
                    S = cv2.contourArea(cont, oriented=True)
                    OCRArea = OCRArea + abs(S)

                    pts = cv2.boxPoints(rect)
                    pts = np.int0(pts).reshape(4,2)
                    pred_bbox.append(pts)
                    
                    text_inst = {}
                    text_inst["lineBox"] = pts.tolist()
                    text_inst["score"] = str(output_dict['confidences'][j])
                    text_inst["contour"] = cont.tolist()
                    lines.append(text_inst)

                min_bounding_rects = []
                min_bounding_area = []
                if self.need_layout:
                    clusters = cluster_polygons(pred_bbox)
                    for cluster in clusters:
                        min_rect, min_area = minimum_bounding_rectangle(cluster)
                        min_bounding_rects.append(min_rect.tolist())
                        min_bounding_area.append(min_area)
                
                
                
                out_puts.append({
                        "area_rateing": OCRArea / (H*W),
                        "text_number": len(lines),
                        "boundingBox": min_bounding_rects,
                        "layout_area_rateing": sum(min_bounding_area)/ (H*W),
                        "lines": lines,
                    })

        return out_puts


if __name__ == "__main__":
    pass
