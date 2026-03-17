from datetime import timedelta
from pathlib import Path

import torch
import numpy as np

from isegm.data.datasets import GrabCutDataset, BerkeleyDataset, DavisDataset, SBDEvaluationDataset, PascalVocDataset, Davis585Dataset, COCOMValDataset, COD10kDataset, CAMODataset, ISTDDataset, TRASHCANDataset

from isegm.utils.serialization import load_model

import torch.nn.functional as F


def masked_bce_loss(output, gt):
    """
    output: (H, W) - model output logits (0~1)
    gt: (H, W) - ground truth labels (-1: ignore, 0: background, 1: foreground)
    """
    valid_mask = gt != -1  # -1인 부분을 제외
    pos_mask = gt == 1     # foreground 부분
    neg_mask = gt == 0     # background 부분

    # BCE loss 적용 (pos_mask만 적용)
    bce_loss = F.binary_cross_entropy(output[pos_mask], gt[pos_mask].float(), reduction='none')

    # 배경 부분 (gt==0)은 0이 되도록 loss 적용
    background_loss = F.binary_cross_entropy(output[neg_mask], torch.zeros_like(output[neg_mask]), reduction='none')

    # 전체 loss (valid한 영역에 대해서만)
    total_loss = torch.zeros_like(output)
    total_loss[pos_mask] = bce_loss
    total_loss[neg_mask] = background_loss

    return total_loss[valid_mask].mean()  # -1이 아닌 부분에 대해 평균 loss 반환

def masked_bce_loss_v2(output, gt):
    """
    output: (H, W) - model output logits (0~1)
    gt: (H, W) - ground truth labels (-1: ignore, 0: background, 1: foreground)
    """
    valid_mask = gt != -1  # -1인 부분을 제외
    pos_mask = gt == 1     # foreground 부분
    neg_mask = gt == 0     # background 부분

    # BCE loss 적용 (pos_mask만 적용)
    bce_loss = F.binary_cross_entropy(output[pos_mask], gt[pos_mask].float(), reduction='none')

    # 배경 부분 (gt==0)은 0이 되도록 loss 적용
    background_loss = F.binary_cross_entropy(output[neg_mask], torch.zeros_like(output[neg_mask]), reduction='none')

    # 전체 loss (valid한 영역에 대해서만)
    total_loss = torch.zeros_like(output)
    total_loss[pos_mask] = bce_loss
    total_loss[neg_mask] = background_loss

    return total_loss[pos_mask].mean()+total_loss[neg_mask].mean()
    # return total_loss[valid_mask].mean()  # -1이 아닌 부분에 대해 평균 loss 반환

def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    #print(state_dict['config'], **kwargs )
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='val')
    elif dataset_name == 'COCO_MVal':
        dataset = COCOMValDataset(cfg.COCO_MVAL_PATH)
    elif dataset_name == 'D585_SP':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='sp')
    elif dataset_name == 'D585_STM':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='stm')
    elif dataset_name == 'D585_ZERO':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='zero')
    elif dataset_name == 'COD10k':
        dataset = COD10kDataset(cfg.COD10K_PATH)
    elif dataset_name == 'CAMO':
        dataset = CAMODataset(cfg.CAMO_PATH)
    elif dataset_name == 'ISTD':
        dataset = ISTDDataset(cfg.ISTD_PATH)
    elif dataset_name == 'TRASHCAN':
        # import pdb;pdb.set_trace()
        dataset = TRASHCANDataset(cfg.TRASHCAN_PATH)
    else:
        dataset = None
    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=int)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        # import pdb;pdb.set_trace()
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None):
    table_header = (f'|{"Pipeline":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row
