from time import time

import pdb

import numpy as np
import torch
import os
from isegm.inference import utils
from isegm.inference.clicker import Clicker
import shutil
import cv2
from isegm.utils.vis import add_tag

import matplotlib.pyplot as plt
import copy
import torch.nn as nn

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, args=None, vis = True, vis_path = './experiments/vis_val/',logs_path = None, writer=None, **kwargs):

    all_ious = []
    all_first_mask = []
    if vis:
        save_dir =  vis_path + dataset.name +str(logs_path).split('/')[-1] + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        save_dir = None

    for name, param in predictor.net.model.named_parameters():
        if 'mask_decoder' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    sam_model_copy = copy.deepcopy(predictor.net.model)

    start_time = time()
    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()

    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        
        predictor.net.model = copy.deepcopy(sam_model_copy)       
        _, sample_ious, _, max_mask = evaluate_sample(sample.image, sample.gt_mask, sample.init_mask, predictor,
                                            sample_id=dataset.dataset_samples[index], vis= vis, save_dir = save_dir,
                                            index = index, writer=writer, **kwargs)
        all_ious.append(sample_ious)
        all_first_mask.append(max_mask)
        
        noc = len(sample_ious)
        noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)

        writer.add_scalar("Th90/NoC_per_sample", noc, index)
        
        writer.add_scalar("Th80/NoC", noc_list[0], index)
        writer.add_scalar("Th85/NoC", noc_list[1], index)
        writer.add_scalar("Th90/NoC", noc_list[2], index)

        writer.add_scalar("Th80/FR", over_max_list[0], index)
        writer.add_scalar("Th85/FR", over_max_list[1], index)
        writer.add_scalar("Th90/FR", over_max_list[2], index)
        
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

def Progressive_Merge(pred_mask, previous_mask, y, x):
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    num, labels = cv2.connectedComponents(diff_regions.astype(np.uint8))
    label = labels[y,x]
    corr_mask = labels == label
    if previous_mask[y,x] == 1:
        progressive_mask = np.logical_and( previous_mask, np.logical_not(corr_mask))
    else:
        progressive_mask = np.logical_or( previous_mask, corr_mask)
    return progressive_mask


def evaluate_sample(image, gt_mask, init_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, vis = True, save_dir = None, index = 0, callback=None,
                    progressive_mode = True, writer=None
                    ):
    
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    prev_mask = None
    max_mask = None
    ious_list = []

    predictor.set_input_image(image)
    predictor.mask_idx = None
    predictor.prev_logit = None

    num_pm = 999
    if init_mask is not None:
        predictor.set_prev_mask(init_mask)
        pred_mask = init_mask
        prev_mask = init_mask
        num_pm = 0
        
    for idx_click in range(max_clicks):

        # Simulate human oracle (click)
        clicker.make_next_click(pred_mask)

        # Get mask using prev_mask and current_click
        with torch.no_grad():
            pred_mask, logits = predictor.get_prediction(clicker, prev_mask, mode='eval')

        # Visualize on tensorboard
        if index<30:
            clicks_curr = [[click.coords[0],click.coords[1],click.is_positive*1] for click in clicker.clicks_list]
            clicks_curr = np.asarray(clicks_curr)
            clicks_pos = clicks_curr[clicks_curr[:,2]==1][:,:2]
            clicks_neg = clicks_curr[clicks_curr[:,2]==0][:,:2]

            pred_mask_vis = pred_mask.unsqueeze(0).repeat(3,1,1)*1.
            pred_mask_vis = visualize_click(pred_mask_vis, clicks_pos, [0,1,0])
            pred_mask_vis = visualize_click(pred_mask_vis, clicks_neg, [1,0,0])
            writer.add_image(sample_id[:-4], pred_mask_vis, idx_click)
        
        pred_mask = pred_mask.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()

        if progressive_mode:
            clicks = clicker.get_clicks()
            if len(clicks) >= num_pm: 
                last_click = clicks[-1]
                last_y, last_x = last_click.coords[0], last_click.coords[1]
                pred_mask = Progressive_Merge(pred_mask, prev_mask, last_y, last_x)
                predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask,0),0)
        
        if callback is not None:
            callback(image, gt_mask, pred_mask, sample_id, idx_click, clicker.clicks_list)

        iou = utils.get_iou(gt_mask, pred_mask)
        ious_list.append(iou)
        prev_mask = pred_mask

        if iou >= max_iou_thr and idx_click + 1 >= min_clicks:
            break

    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_mask, max_mask


def visualize_click(img, clicks, color, square_half_size=4):

    height, width = img.shape[1], img.shape[2]

    for y, x in clicks:
        # Define the square boundaries
        y_min = max(0, y - square_half_size)
        y_max = min(height, y + square_half_size + 1)
        x_min = max(0, x - square_half_size)
        x_max = min(width, x + square_half_size + 1)

        # Draw the red square
        img[0, y_min:y_max, x_min:x_max] = color[0]
        img[1, y_min:y_max, x_min:x_max] = color[1]
        img[2, y_min:y_max, x_min:x_max] = color[2]

    return img