import os, glob, time, argparse, datetime, shutil
from functools import cmp_to_key
import fitz, cv2
import numpy as np
import networkx as nx
from PIL import Image
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.ocr_utils import Painter, enlarge_rects, boxes_to_rects, points_in_rects
from modules.ocr_utils import rects_inte_rects, rects_union_rects, rects_area
from modules.ocr_utils import split_interval_excluding, get_cropped_image, extend_lists, clean_text
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import unimernet
from unimernet.common.config import Config as unimernet_Config
from surya.model.ordering.processor import load_processor as surya_load_processor
from surya.model.ordering.model import load_model as surya_load_model
from surya.ordering import batch_ordering
from rapid_table import RapidTable, VisTable
from rapidocr_paddle import RapidOCR 


def layout_model_init(weight, config_file):
    model = Layoutlmv3_Predictor(weight, config_file)
    return model


def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model


def mfr_model_init(weight_dir, device='cpu'):
    args = argparse.Namespace(cfg_path="configs/unimernet_config.yaml", options=None)
    cfg = unimernet_Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "unimernet_base.pth")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = unimernet.tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = unimernet.processors.load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


def mfd(mfd_model, img, img_size=1888, conf_thres=0.25, iou_thres=0.45):
    mfd_res = mfd_model.predict(img, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0] 
    mf_dets = [] 
    for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()): 
        xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy] 
        mf_item = { 
            'category_id': 13 + int(cla.item()),
            'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
            'score': round(float(conf.item()), 2),
            }
        mf_dets.append(mf_item)
    mf_dets = np.array(mf_dets, object)
    mf_boxes = np.array([mf_det['poly'] for mf_det in mf_dets], int).reshape([-1, 8])
    mf_rects = boxes_to_rects(mf_boxes).reshape([-1, 4])
    mf_points = np.transpose(mf_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    mf_scores = np.array([mf_det['score'] for mf_det in mf_dets])
    mf_cats = np.array([mf_det['category_id'] for mf_det in mf_dets])
    return mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats


def mf_refine(mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats):
    if not len(mf_dets):
        return mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats
    mf_inte_x_min, mf_inte_y_min, mf_inte_x_max, mf_inte_y_max, \
        mf_inte_area, mf_inte_valid = rects_inte_rects(mf_rects, mf_rects, return_valid=True)
    mf_rects_area = rects_area(mf_rects)
    mask_mf_sub = (mf_inte_area / mf_rects_area.reshape([-1, 1])) > 0.8
    mask_mf_dup = mask_mf_sub * mask_mf_sub.T
    mask_mf_sub = mask_mf_sub * (~mask_mf_dup)
    mask_mf_dup = mask_mf_dup * (((mf_scores[:, np.newaxis] - mf_scores) < 0) +\
        (((mf_scores[:, np.newaxis] - mf_scores) == 0) * (np.arange(len(mf_dets))[:, np.newaxis] - np.arange(len(mf_dets)) > 0)))
    inds_mf = ~np.any(mask_mf_dup + mask_mf_sub, axis=1)
    mf_dets = mf_dets[inds_mf]
    mf_boxes = mf_boxes[inds_mf]
    mf_rects = mf_rects[inds_mf]
    mf_points = mf_points[:, inds_mf, :]
    mf_scores = mf_scores[inds_mf]
    mf_cats = mf_cats[inds_mf]
    
    mf_union_rects_xmin, mf_union_rects_ymin, mf_union_rects_xmax, mf_union_rects_ymax, \
        mf_union_rects_area = rects_union_rects(mf_rects, mf_rects)
    mf_union_height = mf_union_rects_ymax - mf_union_rects_ymin
    mf_inte_x_min, mf_inte_y_min, mf_inte_x_max, mf_inte_y_max, \
        mf_inte_area, mf_inte_valid = rects_inte_rects(mf_rects, mf_rects, return_valid=False)
    mf_apart_width = np.where(mf_inte_x_min > mf_inte_x_max, mf_inte_x_min - mf_inte_x_max, 0)
    mf_inte_height = np.where(mf_inte_y_min < mf_inte_y_max, mf_inte_y_max - mf_inte_y_min, 0)
    mf_height = mf_rects[:, 3] - mf_rects[:, 1]
    mf_height = np.minimum(mf_height[:, np.newaxis], mf_height)
    mf_union_mask1 = np.ones([len(mf_cats), len(mf_cats)], bool)
    mf_union_mask1[(mf_cats==14)[:, np.newaxis] & (mf_cats==14).T] = False
    mf_union_mask1[mf_cats==14, mf_cats==14] = True
    mf_union_mask1 = mf_union_mask1 * ((mf_inte_height / mf_union_height) > 0.6) * (mf_apart_width < (mf_height * 0.1))
    mf_union_mask2 = (np.where(mf_inte_valid, mf_inte_area, 0) / mf_union_rects_area) > 0.6
    mf_union_mask = mf_union_mask1 + mf_union_mask2
    inline_union_mf_edges = list(zip(*list(np.where(mf_union_mask))))
    g = nx.Graph()
    g.add_edges_from(inline_union_mf_edges) # pass pairs here
    g_comps = [list(a) for a in list(nx.connected_components(g))] # get merged pairs here
    mf_boxes_merged = []
    mf_scores_merged = []
    mf_cats_merged = [] 
    for g_comp in g_comps:
        mf_cats_merged.append(np.max(mf_cats[g_comp]))
        mf_scores_merged.append(np.min(mf_scores[g_comp]))
        if len(g_comp) > 1:
            mf_rect = [np.min(mf_rects[g_comp][:, 0]), np.min(mf_rects[g_comp][:, 1]), np.max(mf_rects[g_comp][:, 2]), np.max(mf_rects[g_comp][:, 3])]
            mf_box = [mf_rect[0], mf_rect[1], mf_rect[2], mf_rect[1], mf_rect[2], mf_rect[3], mf_rect[0], mf_rect[3]]
            mf_boxes_merged.append(mf_box)      
        else:
            mf_boxes_merged.append(mf_boxes[g_comp[0]])
    mf_boxes = np.array(mf_boxes_merged)
    mf_scores = np.array(mf_scores_merged)
    mf_cats = np.array(mf_cats_merged)
    mf_rects = boxes_to_rects(mf_boxes)
    mf_points = np.transpose(mf_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    return mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats


def layout_det(layout_model, img, min_side_len=800): 
    h, w, _ = img.shape
    factor = np.clip(min_side_len / h if h < w else min_side_len / w, 0, 1)
    img_resized = cv2.resize(img, (round(w*factor), round(h*factor)))
    layout_dets = layout_model(img_resized, ignore_catids=[])['layout_dets'] 
    for det in layout_dets:
        det['poly'] = [c / factor for c in det['poly']]
    return layout_dets


def layout_filter(layout_dets, mf_dets):
    mf_dets_iso = [
        {'category_id': det['category_id'],
         'poly': det['poly'],
         'score': 1,
         } for i, det in enumerate(mf_dets) if det['category_id'] == 14]
    # layout_dets = [det for det in layout_dets if det['category_id'] != 8]
    layout_dets.extend(mf_dets_iso)
    layout_dets = np.array(layout_dets, object)
    layout_boxes = np.array([layout_det['poly'] for layout_det in layout_dets]).reshape([-1, 8])
    layout_rects = boxes_to_rects(layout_boxes)
    layout_points = np.transpose(layout_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    layout_scores = np.array([det['score'] for det in layout_dets])
    layout_cats = np.array([det['category_id'] for det in layout_dets])
    return layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats
    
    
def layout_refine(layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats): 
    layout_inte_x_min, layout_inte_y_min, layout_inte_x_max, layout_inte_y_max, \
        layout_inte_area, layout_inte_valid = rects_inte_rects(layout_rects, layout_rects, return_valid=True)
    layout_rects_area = rects_area(layout_rects)
    mask_layout_sub = (layout_inte_area / layout_rects_area.reshape([-1, 1])) > 0.8
    mask_layout_dup = mask_layout_sub * mask_layout_sub.T
    mask_layout_sub = mask_layout_sub * (~mask_layout_dup)
    mask_layout_sub_fig = np.ones([len(layout_dets), len(layout_dets)], bool)
    fig_inds = (layout_cats==3).nonzero()[0]
    mask_layout_sub_fig[:, fig_inds] = False
    mask_layout_sub_fig[fig_inds, :] = False
    mask_layout_sub_fig[np.ix_(fig_inds, fig_inds)] = True
    mask_layout_sub = mask_layout_sub * mask_layout_sub_fig
    mask_layout_dup = mask_layout_dup * (((layout_scores[:, np.newaxis] - layout_scores) < 0) +\
        (((layout_scores[:, np.newaxis] - layout_scores) == 0) * (np.arange(len(layout_dets))[:, np.newaxis] - np.arange(len(layout_dets)) > 0)))
    inds_layout = ~np.any(mask_layout_dup + mask_layout_sub, axis=1)
    layout_dets = layout_dets[inds_layout]
    layout_boxes = layout_boxes[inds_layout]
    layout_rects = layout_rects[inds_layout]
    layout_scores = layout_scores[inds_layout]
    layout_cats = layout_cats[inds_layout]
    layout_points = layout_points[:, inds_layout, :]
    return layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats


def layout_order(surya_model, surya_processor, pil_img, layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats):
    layout_order = np.argsort([res['position'] for res in batch_ordering([pil_img], [layout_rects.tolist()], surya_model, surya_processor)[0].dict()['bboxes']]) 
    layout_dets = layout_dets[layout_order] 
    layout_boxes = layout_boxes[layout_order] 
    layout_rects = layout_rects[layout_order] 
    layout_points = layout_points[:, layout_order, :]
    layout_scores = layout_scores[layout_order] 
    layout_cats = layout_cats[layout_order]   
    return layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats


def ptd(pt_model, img): 
    pt_boxes = pt_model.text_det(img)[0].reshape([-1, 8]) 
    pt_rects = boxes_to_rects(pt_boxes)
    return pt_boxes, pt_rects


def pt_split(pt_boxes, pt_rects, mf_rects): 
    mf_inte_pt_xmin, mf_inte_pt_ymin, mf_inte_pt_xmax, mf_inte_pt_ymax, mf_inte_pt_area, mask_mf_inte_pt = \
        rects_inte_rects(mf_rects, pt_rects)   
    mf_height, pt_height = mf_rects[:, 3] - mf_rects[:, 1], pt_rects[:, 3] - pt_rects[:, 1]
    mask_mf_break_pt = ((mf_inte_pt_ymax - mf_inte_pt_ymin) / np.minimum(mf_height[:, np.newaxis], pt_height)) > 0.5
    mf_break_inds, pt_break_inds = np.where(mask_mf_break_pt)
    pt_boxes_splited = np.expand_dims(pt_boxes, 1).tolist()
    for pt_ind in np.unique(pt_break_inds):
        pt_x = pt_rects[pt_ind][[0, 2]]
        pt_y = pt_rects[pt_ind][[1, 3]]
        mf_inds = mf_break_inds[pt_break_inds==pt_ind]
        mf_x = mf_rects[mf_inds][:, [0, 2]]
        pt_x_splited = split_interval_excluding(pt_x, mf_x)
        pt_rect = np.hstack([pt_x_splited[:, 0].reshape(-1, 1), np.repeat(pt_y[0], len(pt_x_splited)).reshape([-1, 1]),\
                              pt_x_splited[:, 1].reshape(-1, 1), np.repeat(pt_y[1], len(pt_x_splited)).reshape([-1, 1])])
        pt_box = np.array([pt_rect[:, 0], pt_rect[:, 1], pt_rect[:, 2], pt_rect[:, 1], pt_rect[:, 2], pt_rect[:, 3], pt_rect[:, 0], pt_rect[:, 3]]).transpose().tolist()
        pt_boxes_splited[pt_ind] = pt_box
    pt_boxes = np.array(extend_lists(pt_boxes_splited), int).reshape([-1, 8])
    pt_points = np.transpose(pt_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    pt_rects = boxes_to_rects(pt_boxes)
    return pt_boxes, pt_rects, pt_points


def pt_att_layout(pt_rects, pt_points, layout_rects):
    mask_pt_in_layout = np.ones([len(pt_rects), len(layout_rects)], bool)
    for pt_points1 in pt_points:
        mask_pt_in_layout = mask_pt_in_layout * points_in_rects(enlarge_rects(layout_rects), pt_points1)
    pt_inte_layout_xmin, pt_inte_layout_ymin, pt_inte_layout_xmax, pt_inte_layout_ymax, pt_inte_layout_area, mask_pt_inte_layout = \
        rects_inte_rects(pt_rects, layout_rects)
    pt_rects_area = rects_area(pt_rects) 
    pt_inte_layout_ratio = pt_inte_layout_area / pt_rects_area.reshape([-1, 1]) 
    mask_pt_att_layout = (pt_inte_layout_ratio > 0.7) + mask_pt_in_layout
    return mask_pt_att_layout


def mf_att_layout(mf_rects, mf_points, layout_rects):
    mask_mf_in_layout = np.ones([len(mf_rects), len(layout_rects)], bool)
    for mf_points1 in mf_points:
        mask_mf_in_layout = mask_mf_in_layout * points_in_rects(enlarge_rects(layout_rects), mf_points1)
    mf_inte_layout_xmin, mf_inte_layout_ymin, mf_inte_layout_xmax, mf_inte_layout_ymax, mf_inte_layout_area, mask_mf_inte_layout = \
        rects_inte_rects(mf_rects, layout_rects)
    mf_rects_area = rects_area(mf_rects)

    mf_inte_layout_ratio = mf_inte_layout_area / mf_rects_area.reshape([-1, 1])
    mask_mf_att_layout = (mf_inte_layout_ratio > 0.7) + mask_mf_in_layout 
    return mask_mf_att_layout


def ptr(pt_model, img, pt_rects, pt_points): 
    rects = pt_rects.copy()
    points = pt_points.copy()
    # cropped_img_width = np.max([np.linalg.norm(pt_points[0] - pt_points[1], axis=1), np.linalg.norm(pt_points[2] - pt_points[3], axis=1)], axis=0).astype(int) 
    cropped_img_height = np.max([np.linalg.norm(pt_points[0] - pt_points[3], axis=1), np.linalg.norm(pt_points[1] - pt_points[2], axis=1)], axis=0).astype(int) 
    cropped_img_factor = np.clip(48/cropped_img_height, 0, 1) 
    valid = np.full(len(pt_rects), True)
    cropped_imgs = []
    for i in range(len(rects)): 
        xmin, ymin, xmax, ymax = rects[i]
        xmin, ymin = np.floor([xmin, ymin]).astype(int)
        xmax, ymax = np.ceil([xmax, ymax]).astype(int)
        pts = points[:, i, :]
        pts[:, 0] = pts[:, 0] - xmin
        pts[:, 1] = pts[:, 1] - ymin
        new_size = [round((xmax-xmin+1)*cropped_img_factor[i]), round((ymax-ymin+1)*cropped_img_factor[i])]
        if not all(new_size):
            valid[i] = False
        else:
            temp_img = cv2.resize(img[ymin:(ymax+1), xmin:(xmax+1)], new_size)
            pts = pts * cropped_img_factor[i]
            pts[:, 0] = np.floor(pts[:, 0])
            pts[:, 1] = np.ceil(pts[:, 1])
            cropped_img = get_cropped_image(temp_img, pts.astype(np.float32))
            cropped_imgs.append(cropped_img)
    texts, scores = tuple(map(list, list(zip(*pt_model.text_rec(cropped_imgs)[0])))) if len(cropped_imgs) else ([], [])
    ptr_texts = np.full(len(pt_rects), '', object)
    ptr_texts[valid] = texts
    ptr_scores = np.full(len(pt_rects), 0, np.float32)
    ptr_scores[valid] = scores
    return ptr_texts, ptr_scores
 
    
def mfr(mfr_model, mfr_transform, device, img, mf_rects, mf_points, mf_cats): 
    rects = mf_rects.copy()
    points = mf_points.copy()
    # cropped_img_width = np.max([np.linalg.norm(mf_points[0] - mf_points[1], axis=1), np.linalg.norm(mf_points[2] - mf_points[3], axis=1)], axis=0).astype(int) 
    cropped_img_height = np.max([np.linalg.norm(mf_points[0] - mf_points[3], axis=1), np.linalg.norm(mf_points[1] - mf_points[2], axis=1)], axis=0).astype(int) 
    cropped_img_factor = np.clip(48/cropped_img_height, 0, 1) 
    #cropped_img_factor = np.full(cropped_img_height.shape, 1)
    cropped_img_factor = np.where(mf_cats == 13, cropped_img_factor, 1)
    valid = np.full(len(mf_rects), True)
    cropped_imgs = []
    for i in range(len(mf_rects)): 
        xmin, ymin, xmax, ymax = rects[i]
        xmin, ymin = np.floor([xmin, ymin]).astype(int)
        xmax, ymax = np.ceil([xmax, ymax]).astype(int)
        pts = points[:, i, :]
        pts[:, 0] = pts[:, 0] - xmin
        pts[:, 1] = pts[:, 1] - ymin
        new_size = [round((xmax-xmin+1)*cropped_img_factor[i]), round((ymax-ymin+1)*cropped_img_factor[i])]
        if not all(new_size):
            valid[i] = False
        else:
            temp_img = cv2.resize(img[ymin:(ymax+1), xmin:(xmax+1)], new_size)
            pts = pts * cropped_img_factor[i]
            pts[:, 0] = np.floor(pts[:, 0])
            pts[:, 1] = np.ceil(pts[:, 1])
            cropped_img = get_cropped_image(temp_img, pts.astype(np.float32))
            cropped_imgs.append(cropped_img)
    cropped_imgs = [Image.fromarray(img) for img in cropped_imgs]
    mfr_dataset = MathDataset(cropped_imgs, transform=mfr_transform)
    mfr_dataloader = DataLoader(mfr_dataset, batch_size=128, num_workers=0)
    texts = [] 
    for imgs in mfr_dataloader: 
        imgs = imgs.to(device)
        texts.extend(mfr_model.generate({'image': imgs})['pred_str'])
    texts = ['${}$'.format(text) if mf_cats[i]==13 else '$${}$$'.format(text) for i, text in enumerate(texts)]
    mfr_texts = np.full(len(mf_rects), '', object)
    mfr_texts[valid] = texts
    return mfr_texts


def ptmf(pt_boxes, pt_rects, pt_points, ptr_texts, ptr_scores, mask_pt_att_layout, mf_boxes, mf_rects, mf_points, mfr_texts, mask_mf_att_layout): 
    ptmf_boxes = np.vstack([pt_boxes, mf_boxes])
    ptmf_rects = np.vstack([pt_rects, mf_rects])
    ptmf_points = np.concatenate([pt_points, mf_points], axis=1)
    ptrmfr_texts = np.hstack([ptr_texts, mfr_texts]).astype(object) 
    ptrmfr_scores = np.hstack([ptr_scores, np.ones(len(mfr_texts))])    
    mask_ptmf_att_layout = np.concatenate([mask_pt_att_layout, mask_mf_att_layout], axis=0)
    return ptmf_boxes, ptmf_rects, ptmf_points, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout
  
    
def layout_ptmf(layout_dets, ptmf_boxes, ptmf_rects, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout):
    ptmf_inte_x_min, ptmf_inte_y_min, ptmf_inte_x_max, ptmf_inte_y_max, \
        ptmf_inte_area, ptmf_inte_valid = rects_inte_rects(ptmf_rects, ptmf_rects, return_valid=False) 
    ptmf_inte_height = np.where(ptmf_inte_y_min < ptmf_inte_y_max, ptmf_inte_y_max - ptmf_inte_y_min, 0) 
    ptmf_height = ptmf_rects[:, 3] - ptmf_rects[:, 1] 
    mask_ptmf_inte = (ptmf_inte_height / np.minimum(ptmf_height[:, np.newaxis], ptmf_height)) > 0.5 
    ptmf_order = np.where(mask_ptmf_inte, ptmf_rects[:, 2][:, np.newaxis] - ptmf_rects[:, 2], ptmf_rects[:, 3][:, np.newaxis] - ptmf_rects[:, 3]) 
    for layout_ind, det in enumerate(layout_dets):
        ptmf_inds = mask_ptmf_att_layout[:, layout_ind].nonzero()[0]
        layout_ptmf_order = sorted(range(len(ptmf_inds)), key=lambda i: cmp_to_key(lambda x, y: ptmf_order[x, y])(ptmf_inds[i]))
        ptmf_inds = ptmf_inds[layout_ptmf_order]
        det['ptmf_boxes'] = ptmf_boxes[ptmf_inds]
        det['ptmf_rects'] = ptmf_rects[ptmf_inds]
        det['ptrmfr_texts'] = ptrmfr_texts[ptmf_inds]
        det['ptrmfr_scores'] = ptrmfr_scores[ptmf_inds]
    return


def layout_figure(pil_img, layout_dets, layout_rects, output_dir, upload_dir, figure_offset, page_id):
    fig_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] == 3]
    figcap_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] == 4 or 
                   (det['category_id'] == 1 and len(det['ptrmfr_texts']) and det['ptrmfr_texts'][0].lower().startswith('fig'))]
    fig_inds, fig_dets = map(list, list(zip(*fig_dets))) if len(fig_dets) else ([], [])
    figcap_inds, figcap_dets = map(list, list(zip(*figcap_dets))) if len(figcap_dets) else ([], [])
    figcap_adj_num = [2 if det['category_id'] == 4 else 1 for det in figcap_dets]
    mask_fig_adj_cap = (np.abs(np.array(fig_inds)[:, np.newaxis] - np.array(figcap_inds))) <= np.array(figcap_adj_num)[np.newaxis, :]
    fig_rects, figcap_rects = layout_rects[fig_inds], layout_rects[figcap_inds]
    fig_inte_cap_x_min, fig_inte_cap_y_min, fig_inte_cap_x_max, fig_inte_cap_y_max, \
        fig_inte_cap_area, fig_inte_cap_valid = rects_inte_rects(fig_rects, figcap_rects, return_valid=False)
    fig_width, fig_height = fig_rects[:, 2] - fig_rects[:, 0], fig_rects[:, 3] - fig_rects[:, 1]
    figcap_width, figcap_height = figcap_rects[:, 2] - figcap_rects[:, 0], figcap_rects[:, 3] - figcap_rects[:, 1]
    mask_fig_inte_cap_x = ((fig_inte_cap_x_max - fig_inte_cap_x_min) / np.minimum(fig_width[:, np.newaxis], figcap_width)) > 0.7
    mask_fig_inte_cap_y = ((fig_inte_cap_y_max - fig_inte_cap_y_min) / np.minimum(fig_height[:, np.newaxis], figcap_height)) > 0.7
    dist_fig_cap = np.full(mask_fig_inte_cap_x.shape, np.inf)
    dist_fig_cap = np.where(mask_fig_inte_cap_x, fig_inte_cap_y_min - fig_inte_cap_y_max, dist_fig_cap)
    dist_fig_cap = np.where(mask_fig_inte_cap_y, fig_inte_cap_x_min - fig_inte_cap_x_max, dist_fig_cap)
    dist_fig_cap = np.where(mask_fig_inte_cap_x * mask_fig_inte_cap_y, 0, dist_fig_cap)
    dist_fig_cap[~mask_fig_adj_cap] = np.inf
    if all(dist_fig_cap.shape): 
        inds_cap_att_fig = np.argmin(dist_fig_cap, axis=0)  
        inds_cap_att_fig = np.where(dist_fig_cap[inds_cap_att_fig, np.arange(dist_fig_cap.shape[1])] < np.inf, inds_cap_att_fig, None)
    else:
        inds_cap_att_fig = np.full(dist_fig_cap.shape[1], None)      
    valid = inds_cap_att_fig != None
    figcap_dets = [det for i_figcap, det in enumerate(figcap_dets) if valid[i_figcap]]
    inds_cap_att_fig, figcap_rects = inds_cap_att_fig[valid], figcap_rects[valid]
    figcap_width, figcap_height = figcap_width[valid], figcap_height[valid]
    
    figures = []
    for i_fig, det in enumerate(fig_dets):
        xmin, ymin, xmax, ymax = np.round(fig_rects[i_fig]).astype(int)
        cropped_img = pil_img.crop([xmin, ymin, xmax, ymax])
        url = 'figures/{}-{}-whxyp_{}_{}_{}_{}_{}.jpg'.format(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
            figure_offset + i_fig + 1,
            round(fig_width[i_fig]/pil_img.width, 4),
            round(fig_height[i_fig]/pil_img.height, 4), 
            round(fig_rects[i_fig, 0]/pil_img.width, 4), 
            round(fig_rects[i_fig, 1]/pil_img.height, 4), 
            int(page_id),
            )                                                    
        cropped_img.save(os.path.join(output_dir, url))
        det['url'] = url   
        #utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        figures.append({'location': [[xmin, ymin, xmax, ymax], int(page_id)],
                        'oss_path': os.path.join(upload_dir, url),
                        'figure_id': figure_offset + i_fig + 1,
                        'caption': []
                        })  
    for i_figcap, det in enumerate(figcap_dets):
        content = clean_text(' '.join(det['ptrmfr_texts']))
        url = fig_dets[inds_cap_att_fig[i_figcap]]['url'].split('-whxyp')[0]
        url = '{}-whxyp_{}_{}_{}_{}_{}.txt'.format(
            url, 
            round(figcap_width[i_figcap]/pil_img.width, 4),
            round(figcap_height[i_figcap]/pil_img.height, 4),
            round(figcap_rects[i_figcap, 0]/pil_img.width, 4), 
            round(figcap_rects[i_figcap, 1]/pil_img.height, 4),
            int(page_id))                                                
        with open(os.path.join(output_dir, url), 'w', encoding='utf-8') as f:
            f.write(content)
        # utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        figures[inds_cap_att_fig[i_figcap]]['caption'].append(content)
    return figures
    

def layout_table(table_model, img, layout_dets, layout_rects, output_dir, upload_dir, table_offset, page_id):
    tab_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] == 5]
    tabcap_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] in {6, 7} or 
                   (det['category_id'] == 1 and len(det['ptrmfr_texts']) and det['ptrmfr_texts'][0].lower().startswith('table'))]
    tab_inds, tab_dets = map(list, list(zip(*tab_dets))) if len(tab_dets) else ([], [])
    tabcap_inds, tabcap_dets = map(list, list(zip(*tabcap_dets))) if len(tabcap_dets) else ([], [])
    tabcap_adj_num = [2 if det['category_id'] == 6 else 1 for det in tabcap_dets]
    mask_tab_adj_cap = (np.abs(np.array(tab_inds)[:, np.newaxis] - np.array(tabcap_inds))) <= np.array(tabcap_adj_num)[np.newaxis, :]
    tab_rects, tabcap_rects = layout_rects[tab_inds], layout_rects[tabcap_inds]
    tab_inte_cap_x_min, tab_inte_cap_y_min, tab_inte_cap_x_max, tab_inte_cap_y_max, \
        tab_inte_cap_area, tab_inte_cap_valid = rects_inte_rects(tab_rects, tabcap_rects, return_valid=False)
    tab_width, tab_height = tab_rects[:, 2] - tab_rects[:, 0], tab_rects[:, 3] - tab_rects[:, 1]
    tabcap_width, tabcap_height = tabcap_rects[:, 2] - tabcap_rects[:, 0], tabcap_rects[:, 3] - tabcap_rects[:, 1]
    mask_tab_inte_cap_x = ((tab_inte_cap_x_max - tab_inte_cap_x_min) / np.minimum(tab_width[:, np.newaxis], tabcap_width)) > 0.7
    mask_tab_inte_cap_y = ((tab_inte_cap_y_max - tab_inte_cap_y_min) / np.minimum(tab_height[:, np.newaxis], tabcap_height)) > 0.7
    dist_tab_cap = np.full(mask_tab_inte_cap_x.shape, np.inf)
    dist_tab_cap = np.where(mask_tab_inte_cap_x, tab_inte_cap_y_min - tab_inte_cap_y_max, dist_tab_cap)
    dist_tab_cap = np.where(mask_tab_inte_cap_y, tab_inte_cap_x_min - tab_inte_cap_x_max, dist_tab_cap)
    dist_tab_cap = np.where(mask_tab_inte_cap_x * mask_tab_inte_cap_y, 0, dist_tab_cap)
    dist_tab_cap[~mask_tab_adj_cap] = np.inf
    if all(dist_tab_cap.shape): 
        inds_cap_att_tab = np.argmin(dist_tab_cap, axis=0)  
        inds_cap_att_tab = np.where(dist_tab_cap[inds_cap_att_tab, np.arange(dist_tab_cap.shape[1])] < np.inf, inds_cap_att_tab, None)
    else:
        inds_cap_att_tab = np.full(dist_tab_cap.shape[1], None)      
    valid = inds_cap_att_tab != None
    tabcap_dets = [det for i_tabcap, det in enumerate(tabcap_dets) if valid[i_tabcap]]
    inds_cap_att_tab, tabcap_rects = inds_cap_att_tab[valid], tabcap_rects[valid]
    tabcap_width, tabcap_height = tabcap_width[valid], tabcap_height[valid]

    tabcap_dets = [det for i_tabcap, det in enumerate(tabcap_dets) if inds_cap_att_tab[i_tabcap] != None]
    tables = []
    for i_tab, det in enumerate(tab_dets): 
        xmin, ymin, xmax, ymax = np.round(tab_rects[i_tab]).astype(int)
        cropped_img = img[ymin:ymax, xmin:xmax]
        tab_ptmf_boxes = det['ptmf_boxes'].copy()
        tab_ptmf_boxes[:, [0, 2, 4, 6]] -= xmin
        tab_ptmf_boxes[:, [1, 3, 5, 7]] -= ymin
        tab_ptmf_result = list(map(list, list(zip(*[tab_ptmf_boxes.reshape([-1, 4, 2]).tolist(), det['ptrmfr_texts'], np.ones(len(det['ptmf_boxes']), np.float32)]))))
        tab_html_str, tab_cell_bboxes, elapsed = table_model(cropped_img, tab_ptmf_result)
        url = 'tables/{}-{}-whxyp_{}_{}_{}_{}_{}.html'.format(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
            table_offset + i_tab + 1,
            round(tab_width[i_tab]/img.shape[1], 4),
            round(tab_height[i_tab]/img.shape[0], 4), 
            round(tab_rects[i_tab, 0]/img.shape[1], 4), 
            round(tab_rects[i_tab, 1]/img.shape[0], 4),
            int(page_id))                                    
        with open(os.path.join(output_dir, url), 'w', encoding='utf-8') as f:
            f.write(tab_html_str)
        det['url'] = url 
        det['html'] = tab_html_str
        #utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        tables.append({'location': [[xmin, ymin, xmax, ymax], int(page_id)],
                        'oss_path': os.path.join(upload_dir, url),
                        'table_id': table_offset + i_tab + 1,
                        'caption': []
                        })  
    for i_tabcap, det in enumerate(tabcap_dets):
        content = clean_text(' '.join(det['ptrmfr_texts']))
        url = tab_dets[inds_cap_att_tab[i_tabcap]]['url'].split('-whxyp')[0]
        url = '{}-whxyp_{}_{}_{}_{}_{}.txt'.format(
            url, 
            round(tabcap_width[i_tabcap]/img.shape[1], 4),
            round(tabcap_height[i_tabcap]/img.shape[0], 4),
            round(tabcap_rects[i_tabcap, 0]/img.shape[1], 4),
            round(tabcap_rects[i_tabcap, 1]/img.shape[0], 4),
            int(page_id))                                                
        with open(os.path.join(output_dir, url), 'w', encoding='utf-8') as f:
            f.write(content)
        # utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        tables[inds_cap_att_tab[i_tabcap]]['caption'].append(content)
    return tables


def layout_content(layout_dets, output_dir, page_id, cat2names): 
    content = ''
    for layout_ind, det in enumerate(layout_dets):
        if cat2names[det['category_id']] in ["title"]:
            det['content'] = clean_text(' '.join(det['ptrmfr_texts']))
            content += '# {}\n\n'.format(det['content'])
        elif cat2names[det['category_id']] in ["plain_text", "isolate_formula", "isolated_formula", "figure_caption", "table_caption", "formula_caption", "table_footnote"]:
            det['content'] = clean_text(' '.join(det['ptrmfr_texts']))
            content += '{}\n\n'.format(det['content'])
        elif cat2names[det['category_id']] in ["figure"]:
            content += '![]({})\n\n'.format(det['url'])
        elif cat2names[det['category_id']] in ["table"]:
            content += '![]({})\n\n'.format(det['url'])
            #content += '{}\n\n'.format(det['html'])
    with open(os.path.join(output_dir, 'debug/{}.md'.format(page_id)), 'w', encoding='utf-8') as f:
        f.write(content)
    return content


def debug(pil_img, layout_dets, layout_rects, pt_boxes, mf_boxes, output_dir, page_id, cat2names):
    s = np.where(pil_img.width > pil_img.height, 3000/pil_img.width, 3000/pil_img.height)
    font_size = max(np.mean(pt_boxes[:, 5]*s - pt_boxes[:, 1]*s), 10)
    width = max(font_size/15, 3) 
    painter = Painter()
    painter.create('temp', pil_img.resize([round(pil_img.width*s), round(pil_img.height*s)]))
    for i, det in enumerate(layout_dets): 
        painter.rectangle('temp', layout_rects[i]*s, 'blue', width)
        painter.text('temp', [layout_rects[i][0]*s, layout_rects[i][1]*s], str(i), 'blue', font_size=font_size*1.5)
        for j, box in enumerate(det['ptmf_boxes']*s):
            painter.polygon('temp', box, 'red', width)
            painter.text('temp', [box[2], box[1]], str(j), 'red', font_size=font_size/1.5)
    painter.save('temp', os.path.join(output_dir, 'debug/{}_a.jpg'.format(page_id)))
    
    painter.create('temp', pil_img.resize([round(pil_img.width*s), round(pil_img.height*s)]))
    for i, rect in enumerate(layout_rects*s): 
        painter.rectangle('temp', rect, 'blue', width)
        painter.text('temp', [rect[0], rect[1]], cat2names[layout_dets[i]['category_id']], 'blue', font_size=font_size*1.5)
    for i, box in enumerate(pt_boxes*s): 
        painter.polygon('temp', box, 'red', width)
        painter.text('temp', [box[2], box[1]], str(i), 'red', font_size=font_size/1.5)
    for i, box in enumerate(mf_boxes*s): 
        painter.polygon('temp', box, 'green', width)
        painter.text('temp', [box[2], box[1]], str(i), 'green', font_size=font_size/1.5)
    painter.save('temp', os.path.join(output_dir, 'debug/{}_b.jpg'.format(page_id)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str)
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--lang', type=str, default="ch")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cuda = device == 'cuda'
    dpi = 600

    layout_model = layout_model_init(weight='models/Layout/model_final.pth', config_file='configs/layoutlmv3_base_inference.yaml')
    surya_model = surya_load_model(checkpoint='models/vikp/surya_order/', device=device)
    surya_processor = surya_load_processor(checkpoint='models/vikp/surya_order/')
    mfd_model = mfd_model_init('models/MFD/weights.pt')     
    mfr_model, mfr_vis_processors = mfr_model_init('models/MFR/unimernet_base', device=device)
    mfr_transform = transforms.Compose([mfr_vis_processors, ])
    pt_model = RapidOCR(config_path='configs/rapidocr_config.yaml', det_use_cuda=use_cuda, rec_use_cuda=use_cuda, 
                          det_model_path='models/PaddleOCR/ch_PP-OCRv3_det_infer',
                          rec_model_path='models/PaddleOCR/ch_PP-OCRv3_rec_slim_infer')
    table_model = RapidTable(model_path='models/RapidTable/ch_ppstructure_mobile_v2_SLANet.onnx')
    cat2names = ["title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption", "table_footnote", 
                "isolate_formula", "formula_caption", " ", " ", " ", "inline_formula", "isolated_formula"]

    if os.path.isdir(args.pdf):
        pdf_paths = glob.glob(os.path.join(args.pdf, '**/*.pdf'), recursive=True)
    else:
        pdf_paths = [args.pdf]
    for i_pdf, pdf_path in enumerate(pdf_paths): 
        output_dir = os.path.join(args.output, os.path.basename(pdf_path).replace('.pdf', ''))
        upload_dir = ''
        print('output to: {}'.format(output_dir))
        if os.path.exists(output_dir):
            shutil.copytree(output_dir, output_dir+'_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"))
            shutil.rmtree(output_dir)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)
        doc = fitz.open(pdf_path)
        doc_content = ''
        doc_figures = []
        doc_tables = []
        for i_page in range(len(doc)): 
            page = doc.load_page(i_page) 
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))  
            pil_img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples) 
            img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR) 
            t1 = time.time()
            mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats = mfd(mfd_model, img) 
            mfd_elapsed = time.time() - t1
            mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats = mf_refine(mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats)
            t1 = time.time()
            layout_dets = layout_det(layout_model, img)
            layout_det_elapsed = time.time() - t1
            layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats = layout_filter(layout_dets, mf_dets)
            layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats = layout_refine(layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats)
            t1 = time.time()
            layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats = layout_order(surya_model, surya_processor, pil_img, layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats)
            layout_order_elapsed = time.time() - t1
            t1 = time.time()
            pt_boxes, pt_rects = ptd(pt_model, img)
            ptd_elapsed = time.time() - t1
            pt_boxes, pt_rects, pt_points = pt_split(pt_boxes, pt_rects, mf_rects) 
            mask_pt_att_layout = pt_att_layout(pt_rects, pt_points, layout_rects) 
            mask_mf_att_layout = mf_att_layout(mf_rects, mf_points, layout_rects)
            t1 = time.time()
            ptr_texts, ptr_scores = ptr(pt_model, img, pt_rects, pt_points)
            ptr_elapsed = time.time() - t1
            t1 = time.time()
            mfr_texts = mfr(mfr_model, mfr_transform, device, img, mf_rects, mf_points, mf_cats)
            mfr_elapsed = time.time() - t1
            ptmf_boxes, ptmf_rects, ptmf_points, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout = ptmf(pt_boxes, pt_rects, pt_points, ptr_texts, ptr_scores, mask_pt_att_layout, mf_boxes, mf_rects, mf_points, mfr_texts, mask_mf_att_layout)
            layout_ptmf(layout_dets, ptmf_boxes, ptmf_rects, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout)
            page_id = str(i_page+1).zfill(len(str(len(doc))))
            figure_offset = len(doc_figures)
            figures = layout_figure(pil_img, layout_dets, layout_rects, output_dir, upload_dir, figure_offset, page_id)
            doc_figures.extend(figures)
            t1 = time.time()
            table_offset = len(doc_tables)
            tables = layout_table(table_model, img, layout_dets, layout_rects, output_dir, upload_dir, table_offset, page_id)
            doc_tables.extend(tables)
            table_elapsed = time.time() - t1
            content = layout_content(layout_dets, output_dir, page_id, cat2names)
            doc_content += content
            debug(pil_img, layout_dets, layout_rects, pt_boxes, mf_boxes, output_dir, page_id, cat2names) 
            print('='*80)
            print('page {}, layout {}, mfd {}, mfr {}, ptd {}, ptr {}, table {}, ordering {}'.format(i_page, layout_det_elapsed, mfd_elapsed, mfr_elapsed, ptd_elapsed, ptr_elapsed, table_elapsed, layout_order_elapsed))
            print("="*80)
        with open(os.path.join(output_dir, '{}.md'.format(os.path.basename(pdf_path).replace('.pdf', ''))), 'w', encoding='utf-8') as f: 
            f.write(doc_content)  