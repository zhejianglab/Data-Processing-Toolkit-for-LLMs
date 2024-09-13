# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:22:48 2024

@author: ZJ
"""
import os, cv2, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Painter(object):
    def __init__(self, 
                 box_color=(255, 0, 0), 
                 text_color=(0, 255, 0),
                 box_width = 2,
                 ): 
        self.box_color = box_color
        self.box_width = box_width 
        self.text_color = text_color
        self.figs = {}
        self.workers = {}
    
    def create(self, name, img):
        if isinstance(img, str):
            img = Image.open(img)
        self.figs[name] = img.copy()
        self.workers[name] = ImageDraw.Draw(self.figs[name])
        
    def rectangle(self, name, box, color='green', width=1):
        if isinstance(box, np.ndarray):
            box = box.reshape(-1).tolist()
        if len(box) == 8:
            xmin, ymin, xmax, ymax = box[0], box[1], box[4], box[5]
        else: 
            xmin, ymin, xmax, ymax = box
        if xmin < xmax and ymin < ymax:
            self.workers[name].rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=width)
            
    def polygon(self, name, poly, color='red', width=1):
        self.workers[name].polygon(list(poly), outline=color, width=width)
        
    def text(self, name, pos, s, color='blue', font_size=5):
        x, y = pos
        font_path = 'simhei.ttf'
        font_size = font_size
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        self.workers[name].text((x, y), s, fill=color, font=font)
        
    def save(self, name, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.figs[name].save(output_path)
        temp = self.figs.pop(name)
        temp = self.workers.pop(name)


def enlarge_rects(rects, left=5, right=5, top=5, bottom=5):
    enlarged_rects = rects.copy()
    enlarged_rects[:, 0] -= left
    enlarged_rects[:, 1] -= top
    enlarged_rects[:, 2] += right
    enlarged_rects[:, 3] += bottom
    return enlarged_rects
    

def boxes_to_rects(boxes):
    xmin = boxes[:, [0, 2, 4, 6]][np.arange(boxes.shape[0]).tolist(), np.argmin(boxes[:, [0, 2, 4, 6]], axis=1)]
    xmax = boxes[:, [0, 2, 4, 6]][np.arange(boxes.shape[0]).tolist(), np.argmax(boxes[:, [0, 2, 4, 6]], axis=1)]
    ymin = boxes[:, [1, 3, 5, 7]][np.arange(boxes.shape[0]).tolist(), np.argmin(boxes[:, [1, 3, 5, 7]], axis=1)]
    ymax = boxes[:, [1, 3, 5, 7]][np.arange(boxes.shape[0]).tolist(), np.argmax(boxes[:, [1, 3, 5, 7]], axis=1)]
    rects = np.array([xmin, ymin, xmax, ymax]).transpose()
    return rects
   
    
def points_in_rects(rects, pts):
    xmin = rects[:, 0][:, np.newaxis]
    ymin = rects[:, 1][:, np.newaxis]
    xmax = rects[:, 2][:, np.newaxis]
    ymax = rects[:, 3][:, np.newaxis]
    x = pts[:, 0][np.newaxis, :]
    y = pts[:, 1][np.newaxis, :]
    inside_x = (x >= xmin) & (x <= xmax)
    inside_y = (y >= ymin) & (y <= ymax)
    inside = inside_x & inside_y
    return inside.T 


def rects_inte_rects(rects1, rects2, return_valid=True):
    x_min = np.maximum(rects1[:, 0, np.newaxis], rects2[:, 0])
    y_min = np.maximum(rects1[:, 1, np.newaxis], rects2[:, 1])
    x_max = np.minimum(rects1[:, 2, np.newaxis], rects2[:, 2])
    y_max = np.minimum(rects1[:, 3, np.newaxis], rects2[:, 3])
    valid = (x_min < x_max) & (y_min < y_max)
    width = x_max - x_min
    height = y_max - y_min
    area = height * width
    if return_valid:
        x_min = np.where(valid, x_min, np.nan)
        y_min = np.where(valid, y_min, np.nan)
        x_max = np.where(valid, x_max, np.nan)
        y_max = np.where(valid, y_max, np.nan)
        area = np.where(valid, area, 0)
    return x_min, y_min, x_max, y_max, area, valid


def rects_area(rects):
    width = rects[..., 2] - rects[..., 0]
    height = rects[..., 3] - rects[..., 1]
    area = width * height 
    return area


def rects_union_rects(rects1, rects2):
    rects1_union_rects2_rects_xmin = np.minimum(rects1[:, 0][:, np.newaxis], rects2[:, 0][np.newaxis, :])
    rects1_union_rects2_rects_xmax = np.maximum(rects1[:, 2][:, np.newaxis], rects2[:, 2][np.newaxis, :])
    rects1_union_rects2_rects_ymin = np.minimum(rects1[:, 1][:, np.newaxis], rects2[:, 1][np.newaxis, :])
    rects1_union_rects2_rects_ymax = np.maximum(rects1[:, 3][:, np.newaxis], rects2[:, 3][np.newaxis, :])
    rects1_union_rects2_rects_area = (rects1_union_rects2_rects_xmax - rects1_union_rects2_rects_xmin) * (rects1_union_rects2_rects_ymax - rects1_union_rects2_rects_ymin)
    return rects1_union_rects2_rects_xmin, rects1_union_rects2_rects_ymin, rects1_union_rects2_rects_xmax, rects1_union_rects2_rects_ymax, \
        rects1_union_rects2_rects_area
  

def split_interval_excluding(a, intervals):
    intervals = np.array(intervals)
    intervals = intervals[np.argsort(intervals[:, 0])]
    a_start, a_end = a
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    valid_starts = np.clip(starts, a_start, a_end)
    valid_ends = np.clip(ends, a_start, a_end)
    split_points = np.unique(np.hstack(([a_start], valid_starts, valid_ends, [a_end])))
    split_intervals = np.column_stack((split_points[:-1], split_points[1:]))
    mask = np.ones(len(split_intervals), dtype=bool)
    for start, end in zip(valid_starts, valid_ends):
        mask &= (split_intervals[:, 1] <= start) | (split_intervals[:, 0] >= end)
    result = split_intervals[mask] 
    return result


def get_cropped_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img = np.clip(dst_img, 0.0, 255.0).astype(np.uint8)
    return dst_img


def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)   
    text = text.replace('- ', '-')
    pattern_before = r'(\s+)([\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b])'
    # 正则表达式匹配汉字和中文标点符号后面的空格
    pattern_after = r'([\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b])(\s+)'

    # 使用re.sub替换匹配到的空格
    text = re.sub(pattern_before, r'\2', text)  # 移除前面的空格
    text = re.sub(pattern_after, r'\1', text)  # 移除后面的空格
    return text.strip() 


def extend_lists(ll):
    l = [e for l in ll for e in l]
    return l


   