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

import torch.nn.functional as F
from isegm.inference.utils import masked_bce_loss
# from isegm.inference.utils import kmeans_clustering
from isegm.model.losses import NormalizedFocalLossSigmoid

def evaluate_dataset(dataset, predictor, args=None, vis = True, vis_path = './experiments/vis_val/',logs_path = None, writer=None, **kwargs):

    idx_tta_global = 0
    all_ious = []
    if vis:
        save_dir =  vis_path + dataset.name +str(logs_path).split('/')[-1] + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        save_dir = None

    for name, param in predictor.net.model.named_parameters():
        if 'mask_decoder' in name or 'prompt_encoder' in name:
        # if 'mask_decoder' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    sam_model_copy = copy.deepcopy(predictor.net.model)

    start_time = time()
    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()

    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        
        if np.sum(sample.gt_mask) == 0:
            # Jump no object image
            print('Jump image: ', index)
            continue

        predictor.net.model = copy.deepcopy(sam_model_copy)       
        _, sample_ious, _, max_mask, idx_tta_global = evaluate_sample(sample.image, sample.gt_mask, sample.init_mask, predictor,
                                            idx_data=dataset.dataset_samples[index], vis= vis, save_dir = save_dir,
                                            idx_img=index, writer=writer, idx_tta_global=idx_tta_global, args=args, **kwargs)
 
        ### Metrics
        all_ious.append(sample_ious)
        
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

def evaluate_sample(image, mask_gt, init_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    idx_data=None, vis = True, save_dir = None, idx_img = 0, callback=None,
                    progressive_mode = True, writer=None, idx_tta_global=0, args=None
                    ):
    
    clicker = Clicker(gt_mask=mask_gt)
    mask_curr = np.zeros_like(mask_gt)
    mask_prev = None
    max_mask = None
    ious_list = []

    predictor.set_input_image(image)
    predictor.mask_idx = None
    predictor.prev_logit = None
    merge_flag = False
    clicker_sets = []
    mask_sets = []
    logit_large_sets = []
    logit_sets = []
    logit_sets_prev = []
    model_sets = []
    optimizer_sets = []
    primitive_model = copy.deepcopy(predictor.net.model)
    allclick_model = copy.deepcopy(predictor.net.model)
    # import pdb;pdb.set_trace()
    
    # For TTA optimization
    allclick_optimizer = torch.optim.AdamW(allclick_model.parameters(), lr = args.tta_lr)
    loss_bce = nn.BCEWithLogitsLoss()
    loss_nfl = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2) 
    
    # For loss selection
    is_l_bce = True
    is_l_nfl = False
    is_click_l_bce = True
    
    # For ablation
    do_tta = args.tta
    do_mm = args.mm
    
    # First click
    clicker.make_next_click(mask_curr)
    
    
    for idx_click in range(max_clicks):
        
        # print(str(idx_click)+" click")
        # predictor.net.model.eval()
        last_click = clicker.get_clicks()[-1]

        if len(clicker_sets) == 0:
            # start = time.time()

            predictor.net.model.eval()
            clicker_sets.append(copy.deepcopy(clicker))
            logit_sets_prev.append(None)
            with torch.no_grad():
                mask_curr, logit, logit_256 = predictor.get_prediction(clicker, mask_prev)
            mask_sets.append(mask_curr.clone().cpu().numpy())
            logit_large_sets.append(logit.cpu().detach().unsqueeze(0).numpy())
            logit_sets.append(logit_256.cpu().detach().unsqueeze(0).numpy())
            model_sets.append([copy.deepcopy(predictor.net.model.prompt_encoder),copy.deepcopy(predictor.net.model.mask_decoder)])

            # end = time.time()
            # print("first click time: "+ str(end-start))
            # optimizer_sets.append(torch.optim.AdamW(model_sets[-1].parameters(), lr = args.tta_lr))
            

        # Negative Click
        elif last_click.is_positive == False:
            if do_tta:
                for i, (partial_clicker,partial_mask) in enumerate(zip(clicker_sets, mask_sets)):
                    
                    model_here = copy.deepcopy(predictor.net.model)
                    model_here.prompt_encoder = model_sets[i][0]
                    model_here.mask_decoder = model_sets[i][1]
                    
                    run_TTA(predictor, model_here, torch.optim.AdamW(model_here.parameters(), lr = args.tta_lr), [last_click], partial_clicker, args,
                            logit_sets[i], logit_sets_prev[i], is_l_bce, is_l_nfl, is_click_l_bce,
                            loss_bce, loss_nfl, masked_bce_loss)
                    
                    model_sets[i][0] = predictor.net.model.prompt_encoder
                    model_sets[i][1] = predictor.net.model.mask_decoder
                
            predictor.net.model.eval()
            logit_sets_prev = copy.deepcopy(logit_sets)
            # For every point cluster, add negative point.
            for i, (partial_clicker,partial_mask) in enumerate(zip(clicker_sets, mask_sets)):
                partial_clicker.add_click(last_click)
                with torch.no_grad():
                    mask_curr, logit, logit_256 = predictor.get_prediction(partial_clicker, logit_sets[i])
                mask_sets[i] = mask_curr.clone().cpu().numpy()
                logit_large_sets[i] = logit.cpu().detach().unsqueeze(0).numpy()
                logit_sets[i] = logit_256.cpu().detach().unsqueeze(0).numpy()
                
        # Positive Click
        else:
            individual_clicker = Clicker(gt_mask=mask_gt)
            individual_clicker.add_click(last_click)
            
            # Add neg clicks
            for click in clicker.get_clicks():
                if not click.is_positive:
                    individual_clicker.add_click(click)

            predictor.net.model = primitive_model
            predictor.net.model.eval()
            with torch.no_grad():
                mask_ind, logit_large_ind, logit_ind = predictor.get_prediction(individual_clicker, None)
                
            
            new_cluster_flag = True
            for i, (partial_clicker,partial_mask) in enumerate(zip(clicker_sets, mask_sets)):
                '''
                If there exist the overlapping mask between exist clicks and new click,
                then add the new click to the corresponding cluster and TTA.
                If not, create a new cluster.
                '''
                
                if np.sum(partial_mask*mask_ind.clone().cpu().numpy())>0:
            
                    if do_tta:
                        model_here = copy.deepcopy(predictor.net.model)
                        model_here.prompt_encoder = model_sets[i][0]
                        model_here.mask_decoder = model_sets[i][1]
                        run_TTA(predictor, model_here, torch.optim.AdamW(model_here.parameters(), lr = args.tta_lr), [last_click], partial_clicker, args,
                            logit_sets[i], logit_sets_prev[i], is_l_bce, is_l_nfl, is_click_l_bce,
                            loss_bce, loss_nfl, masked_bce_loss)
                        
                        model_sets[i][0] = predictor.net.model.prompt_encoder
                        model_sets[i][1] = predictor.net.model.mask_decoder
                            
                    predictor.net.model.eval()
                    logit_sets_prev = copy.deepcopy(logit_sets)
                        
                    partial_clicker.add_click(last_click)
                    with torch.no_grad():
                        mask_curr, logit, logit_256 = predictor.get_prediction(partial_clicker, logit_sets[i])
                    mask_sets[i] = mask_curr.clone().cpu().numpy()
                    logit_large_sets[i] = logit.cpu().detach().unsqueeze(0).numpy()
                    logit_sets[i] = logit_256.cpu().detach().unsqueeze(0).numpy()
                    
                    new_cluster_flag = False
                
            # There is no overlapping mask between exist clicks and new click. New cluster is created.
            if new_cluster_flag:    
                clicker_sets.append(individual_clicker)
                mask_sets.append(mask_ind.clone().cpu().numpy())
                logit_large_sets.append(logit_large_ind.cpu().detach().unsqueeze(0).numpy())
                logit_sets.append(logit_ind.cpu().detach().unsqueeze(0).numpy())
                logit_sets_prev.append(None)
                
                new_cluster_model = copy.deepcopy(primitive_model)
                model_sets.append([new_cluster_model.prompt_encoder, new_cluster_model.mask_decoder])
                # optimizer_sets.append(torch.optim.AdamW(new_cluster_model.parameters(), lr = args.tta_lr))
                
                ######## Neg click TTA ###########
                neg_click_list = []
                for click in clicker.get_clicks():
                    if not click.is_positive:
                        neg_click_list.append(click)
                
                # If there is negative points, do tta
                if do_tta and len(neg_click_list) != 0:
                    temp_only_pos_clicker = Clicker(gt_mask=mask_gt)
                    temp_only_pos_clicker.add_click(last_click)
                    
                    model_here = copy.deepcopy(predictor.net.model)
                    model_here.prompt_encoder = model_sets[-1][0]
                    model_here.mask_decoder = model_sets[-1][1]
                    
                    run_TTA(predictor,model_here, torch.optim.AdamW(model_here.parameters(), lr = args.tta_lr), neg_click_list, temp_only_pos_clicker, args,
                        logit_sets[-1], None, is_l_bce, is_l_nfl, is_click_l_bce,
                        loss_bce, loss_nfl, masked_bce_loss)
                    
                    model_sets[-1][0] = predictor.net.model.prompt_encoder
                    model_sets[-1][1] = predictor.net.model.mask_decoder
                
                predictor.net.model.eval()
                logit_sets_prev = copy.deepcopy(logit_sets)
                
                with torch.no_grad():
                    mask_curr, logit, logit_256 = predictor.get_prediction(clicker_sets[-1], logit_sets[-1])
                mask_sets[-1] = mask_curr.clone().cpu().numpy()
                logit_large_sets[-1] = logit.cpu().detach().unsqueeze(0).numpy()
                logit_sets[-1] = logit_256.cpu().detach().unsqueeze(0).numpy()
                
        # All click model         
        predictor.net.model = allclick_model
        
        if do_tta:
            if idx_click>0:
                run_TTA(predictor, allclick_model, allclick_optimizer, [last_click], clicker_backup, args,
                            mask_prev, mask_prev_backup, is_l_bce, is_l_nfl, is_click_l_bce,
                            loss_bce, loss_nfl, masked_bce_loss)

                allclick_model = predictor.net.model
        
        predictor.net.model.eval()
        with torch.no_grad():
            mask_curr, logit, logit_256 = predictor.get_prediction(clicker, mask_prev)
        mask_curr = mask_curr.cpu().detach().numpy()
        all_click_mask = mask_curr
        
        # mask_curr: all click model mask
        
        ############## allclick Union (parts Unions) ########
        mask_pgt = copy.deepcopy(all_click_mask)
        for partial_mask in mask_sets:
            mask_pgt = mask_pgt + partial_mask
        
        mask_pgt = torch.from_numpy(mask_pgt).cuda()
        
        if do_mm:
            if idx_click>0:
                if len(model_sets) == 1:
                    task_vectors = [
                        TaskVector(primitive_model, allclick_model)
                    ]
                    task_vector_sum = sum(task_vectors)
                    mm_model = task_vector_sum.apply_to(copy.deepcopy(primitive_model), scaling_coef= 0.7)
                else:
                    part_task_vectors = []
                    for cluster_model in model_sets:
                        model_here = copy.deepcopy(predictor.net.model)
                        model_here.prompt_encoder = cluster_model[0]
                        model_here.mask_decoder = cluster_model[1]
                        part_task_vectors.append(TaskVector(primitive_model,model_here))
                        
                    part_task_vector_sum = sum(part_task_vectors)
                    part_merge_model = part_task_vector_sum.apply_to(copy.deepcopy(primitive_model), scaling_coef= 0.7)

                    task_vectors = [
                        TaskVector(primitive_model, part_merge_model), TaskVector(primitive_model, allclick_model)
                    ]
                    task_vector_sum = sum(task_vectors)
                    mm_model = task_vector_sum.apply_to(copy.deepcopy(primitive_model), scaling_coef= 0.7)

                predictor.net.model = mm_model
                optimizer_merge = torch.optim.AdamW(predictor.net.model.parameters(), lr = args.tta_lr)
                
                if do_tta:
                    predictor.net.model.train()
                    for idx_tta in range(args.tta_num_iter):   
                        mask_curr, logit, logit_256 = predictor.get_prediction(clicker, mask_prev)
                                        
                        loss = 0.0
                        if is_l_bce:
                            l_bce = loss_bce(logit, mask_pgt.to(torch.float))
                            loss += l_bce
                        if is_l_nfl:
                            l_nfl = torch.mean(loss_nfl(logit, mask_pgt.to(torch.float)))       # mean the batch losses. (actually 1)
                            loss += l_nfl
                        if is_click_l_bce:
                            click_maps = torch.from_numpy(clicker.get_clicks_map()).to(torch.float).cuda()
                            click_l_bce = masked_bce_loss(F.sigmoid(logit), click_maps)
                            loss += click_l_bce
                                
                        optimizer_merge.zero_grad()
                        loss.backward()
                        optimizer_merge.step()
                
                predictor.net.model.eval()
                with torch.no_grad():
                    mask_curr, logit, logit_256 = predictor.get_prediction(clicker, mask_prev)
                mask_curr = mask_curr.cpu().detach().numpy()
        
        else:
            mask_curr = mask_pgt.clone().cpu().detach().numpy()
        
        
        if do_mm:
            visualize_click_per_sets(image, torch.from_numpy(mask_gt).cuda(), mask_pgt, torch.from_numpy(mask_curr).cuda(), clicker.clicks_list, clicker_sets, mask_sets, logit_large_sets, writer, idx_img, idx_data, idx_click)
        else:
            visualize_click_per_sets(image, torch.from_numpy(mask_gt).cuda(), torch.from_numpy(all_click_mask).cuda(), mask_pgt, clicker.clicks_list, clicker_sets, mask_sets, logit_large_sets, writer, idx_img, idx_data, idx_click)
        
        mask_prev_backup = copy.deepcopy(mask_prev)
        mask_prev = logit_256.cpu().detach().unsqueeze(0).numpy() # (1,256,256)

        # Compute IoU
        iou = utils.get_iou(mask_gt, mask_curr)
        ious_list.append(iou)
        if iou >= max_iou_thr and idx_click + 1 >= min_clicks:
            break

        # Perform new click based on current mask
        clicker_backup = copy.deepcopy(clicker)
        clicker.make_next_click(mask_curr)
    
    writer.add_scalar("click_set_num", len(model_sets), idx_img)
       
    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), mask_curr, max_mask, idx_tta_global

def run_TTA(predictor, model, optimizer, click_list, partial_clicker, args,
            logit_in, logit_in_prev, is_l_bce, is_l_nfl, is_click_l_bce,
            loss_bce, loss_nfl, masked_bce_loss):
    """
    Test Time Augmentation(TTA)를 수행하는 함수입니다.
    
    Parameters:
        predictor: 모델 예측을 수행하는 객체
        model: 모델
        optimizer: 옵티마이저
        click_list: add click들이 있는 list
        partial_clicker: 현재까지의 클릭 정보를 가진 객체
        args: TTA 관련 하이퍼파라미터를 가진 객체 (예: tta_lr, tta_num_iter 등)
        logit: 예측시 사용할 logit 값
        logit_prev: 이전 단계의 logit 값
        is_l_bce: BCE 손실 사용 여부 (boolean)
        is_l_nfl: NFL 손실 사용 여부 (boolean)
        is_click_l_bce: 클릭 BCE 손실 사용 여부 (boolean)
        loss_bce: BCE 손실 함수
        loss_nfl: NFL 손실 함수
        masked_bce_loss: 마스킹된 BCE 손실 함수
        
    """
    # 모델과 옵티마이저 상태 불러오기
    predictor.net.model = model
    
    # 마지막 클릭을 반영한 partial_clicker 생성
    partial_clicker_curr = copy.deepcopy(partial_clicker)
    for click in click_list:
        partial_clicker_curr.add_click(click)
    
    # 예측을 통해 기준 마스크 획득 (평가 모드)
    predictor.net.model.eval()
    with torch.no_grad():
        mask_pgt, _, _ = predictor.get_prediction(partial_clicker_curr, logit_in)
    
    # TTA를 위한 모델 학습 모드 전환
    predictor.net.model.train()
    for idx_tta in range(args.tta_num_iter):
        mask_curr, logit, logit_256 = predictor.get_prediction(partial_clicker, logit_in_prev)
        
        loss = 0.0
        if is_l_bce:
            l_bce = loss_bce(logit, mask_pgt.to(torch.float))
            loss += l_bce
        if is_l_nfl:
            # logit과 mask_pgt의 차원을 맞춰서 NFL 손실 계산 (배치 평균)
            l_nfl = torch.mean(loss_nfl(logit.unsqueeze(0), mask_pgt.unsqueeze(0).to(torch.float)))
            loss += l_nfl
        if is_click_l_bce:
            click_maps = torch.from_numpy(partial_clicker_curr.get_clicks_map()).to(torch.float).to(predictor.device)
            
            click_l_bce = masked_bce_loss(torch.sigmoid(logit), click_maps)
            loss += click_l_bce
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 업데이트된 모델과 옵티마이저의 상태 저장
    predictor.net.model.eval()

def click_on_img(img, clicks, color, square_half_size=4):
    height, width = img.shape[1], img.shape[2]
    for y, x in clicks:
        y_min = max(0, y - square_half_size)
        y_max = min(height, y + square_half_size + 1)
        x_min = max(0, x - square_half_size)
        x_max = min(width, x + square_half_size + 1)
        img[0, y_min:y_max, x_min:x_max] = color[0]
        img[1, y_min:y_max, x_min:x_max] = color[1]
        img[2, y_min:y_max, x_min:x_max] = color[2]
    return img

def visualize_click(pred_mask, clicks, writer, idx_img, idx_data, idx_click):
    if idx_img<500:
        clicks_curr = [[click.coords[0],click.coords[1],click.is_positive*1] for click in clicks]
        clicks_curr = np.asarray(clicks_curr)
        clicks_pos = clicks_curr[clicks_curr[:,2]==1][:,:2]
        clicks_neg = clicks_curr[clicks_curr[:,2]==0][:,:2]

        pred_mask_vis = pred_mask.unsqueeze(0).repeat(3,1,1)*1.
        pred_mask_vis = click_on_img(pred_mask_vis, clicks_pos, [0,1,0])
        pred_mask_vis = click_on_img(pred_mask_vis, clicks_neg, [1,0,0])
        writer.add_image(idx_data[:-4], pred_mask_vis, idx_click)

def visualize_click_compare(pred_mask1, pred_mask2, clicks, writer, idx_img, idx_data, idx_click):
    if idx_img<30:
        clicks_curr = [[click.coords[0],click.coords[1],click.is_positive*1] for click in clicks]
        clicks_curr = np.asarray(clicks_curr)
        clicks_pos = clicks_curr[clicks_curr[:,2]==1][:,:2]
        clicks_neg = clicks_curr[clicks_curr[:,2]==0][:,:2]

        pred_mask1_vis = pred_mask1.unsqueeze(0).repeat(3,1,1)*1.
        pred_mask1_vis = click_on_img(pred_mask1_vis, clicks_pos, [0,1,0])
        pred_mask1_vis = click_on_img(pred_mask1_vis, clicks_neg, [1,0,0])

        pred_mask2_vis = pred_mask2.unsqueeze(0).repeat(3,1,1)*1.
        pred_mask2_vis = click_on_img(pred_mask2_vis, clicks_pos, [0,1,0])
        pred_mask2_vis = click_on_img(pred_mask2_vis, clicks_neg, [1,0,0])

        writer.add_image(idx_data[:-4], torch.cat((pred_mask1_vis, pred_mask2_vis),dim=2), idx_click)
        
def visualize_click_per_sets(image, mask_gt, allclick_mask, final_mask, clicks, clicker_sets, mask_sets, logit_large_sets, writer, idx_img, idx_data, idx_click):
    # if idx_img<1000:
    if False:
        image = torch.from_numpy(image).cuda().permute(2,0,1)
        image = image.type(torch.float32) / 255.
        mask_gt = mask_gt.unsqueeze(0).repeat(3,1,1)*1.
        
        clicks_curr = [[click.coords[0],click.coords[1],click.is_positive.item()*1] for click in clicks]
        clicks_curr = np.asarray(clicks_curr)
        clicks_pos = clicks_curr[clicks_curr[:,2]==1][:,:2]
        clicks_neg = clicks_curr[clicks_curr[:,2]==0][:,:2]

        allclick_mask_vis = allclick_mask.unsqueeze(0).repeat(3,1,1)*1.
        allclick_mask_vis = click_on_img(allclick_mask_vis, clicks_pos, [0,1,0])
        allclick_mask_vis = click_on_img(allclick_mask_vis, clicks_neg, [1,0,0])
        
        final_mask_vis = final_mask.unsqueeze(0).repeat(3,1,1)*1.
        final_mask_vis = click_on_img(final_mask_vis, clicks_pos, [0,1,0])
        final_mask_vis = click_on_img(final_mask_vis, clicks_neg, [1,0,0])
        
        image_gt = torch.cat((image, mask_gt),dim=1)
        preds = torch.cat((allclick_mask_vis, final_mask_vis),dim=1)
        final_vis = torch.cat((image_gt, preds),dim=2)
        
        for i in range(len(clicker_sets)):
            clicker_i = clicker_sets[i]
            mask_i = torch.from_numpy(mask_sets[i]).cuda()
            logit_i = torch.from_numpy(logit_large_sets[i]).cuda()
            
            clicks_obj = [[click.coords[0],click.coords[1],click.is_positive.item()*1] for click in clicker_i.get_clicks()]
            clicks_obj = np.asarray(clicks_obj)
            clicks_obj_pos = clicks_obj[clicks_obj[:,2]==1][:,:2]
            clicks_obj_neg = clicks_obj[clicks_obj[:,2]==0][:,:2]
            
            obj_mask_vis = mask_i.unsqueeze(0).repeat(3,1,1)*1.
            obj_mask_vis = click_on_img(obj_mask_vis, clicks_obj_pos, [0,1,0])
            obj_mask_vis = click_on_img(obj_mask_vis, clicks_obj_neg, [1,0,0])
            
            obj_logit_norm = (logit_i - logit_i.min())/(5-logit_i.min())
            obj_logit_vis = obj_logit_norm.repeat(3,1,1)*1.
            obj_logit_vis = click_on_img(obj_logit_vis, clicks_obj_pos, [0,1,0])
            obj_logit_vis = click_on_img(obj_logit_vis, clicks_obj_neg, [1,0,0])
            
            obj_vis = torch.cat((obj_logit_vis, obj_mask_vis),dim=1)
            
            final_vis = torch.cat((final_vis, obj_vis),dim=2)

        writer.add_image(idx_data[:-4], final_vis, idx_click)
        # writer.add_image(idx_data[0], final_vis, idx_click)     # PascalVOC
            
class TaskVector():
    def __init__(self, pretrained_model=None, finetuned_model=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned model instances.

        This can either be done by passing two model instances (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.
        """
        # import pdb;pdb.set_trace()
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_model is not None and finetuned_model is not None
            with torch.no_grad():
                pretrained_state_dict = pretrained_model.state_dict()
                finetuned_state_dict = finetuned_model.state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue  # 정수형 가중치는 무시
                    if 'image_encoder' in key:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]        
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_model, scaling_coef=1.0):
        """Apply a task vector to a pretrained model instance."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    # print(f'Warning: key {key} is present in the pretrained model but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
