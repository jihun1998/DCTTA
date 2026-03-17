import numpy as np
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, center_of_mass

def safe_log(x, epsilon=1e-10):
    x = np.clip(x, epsilon, None)
    return np.log(x)

class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        # click = self._get_next_click_by_uncertainty(pred_mask)
        self.add_click(click)
    
    def make_next_click_by_uncertainty(self, pred_mask, prev_logit, sample_id):
        assert self.gt_mask is not None
        # click = self._get_next_click(pred_mask)
        click = self._get_next_click_by_uncertainty(pred_mask, prev_logit, sample_id)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def compute_score(self, distance_map, confidence_map, w_d=0.5, w_c=0.5):
        # Normalize distance map and confidence map
        norm_distance = distance_map / (distance_map.max() + 1e-6)
        norm_confidence = confidence_map / (confidence_map.max() + 1e-6)
        # Compute combined score
        score = w_d * norm_distance + w_c * (1 - norm_confidence)
        return score

    def get_top_labels(self, labeled_array, top_n=5):
        # 레이블 값별 픽셀 수 계산
        unique_labels, counts = np.unique(labeled_array, return_counts=True)
        
        # 0 (배경) 제거
        if 0 in unique_labels:
            mask = unique_labels != 0
            unique_labels = unique_labels[mask]
            counts = counts[mask]
        
        # 픽셀 개수 기준으로 정렬
        sorted_indices = np.argsort(counts)[::-1]  # 내림차순 정렬
        top_labels = unique_labels[sorted_indices][:top_n]
        top_counts = counts[sorted_indices][:top_n]
    
        return top_labels, top_counts
        # return list(zip(top_labels, top_counts))

    def find_farthest_point_vectorized(self, existing_points, candidate_points):
           
        distances = np.linalg.norm(existing_points[:, np.newaxis] - candidate_points, axis=2)

        min_distances = distances.min(axis=0)

        # 최소 거리 기준으로 정렬
        sorted_indices = np.argsort(min_distances)[::-1]  # 내림차순 정렬
        sorted_candidates = candidate_points[sorted_indices]
        sorted_distances = min_distances[sorted_indices]

        return sorted_candidates, sorted_distances
    
        
    def _get_next_click_by_uncertainty(self, pred_mask, prev_logit,sample_id, padding=True):
        # import pdb;pdb.set_trace()
        if prev_logit is None:
            fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
            fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

            if padding:
                fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
                fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

            fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
            fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

            if padding:
                fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
                fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

            fn_mask_dt = fn_mask_dt * self.not_clicked_map
            fp_mask_dt = fp_mask_dt * self.not_clicked_map

            fn_max_dist = np.max(fn_mask_dt)
            fp_max_dist = np.max(fp_mask_dt)

            is_positive = fn_max_dist > fp_max_dist
            if is_positive:
                coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
            else:
                coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

            return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))
        else:
            
            prev_prob = 1/(1+np.exp(-prev_logit))
            # prev_prob = 1/(1+np.exp(-abs(prev_logit)))
            # entropy = -prev_prob*np.log(prev_prob)-(1-prev_prob)*np.log(1-prev_prob)
            entropy = -prev_prob*safe_log(prev_prob)-(1-prev_prob)*safe_log(1-prev_prob)
            entropy = (entropy-entropy.min())/(entropy.max()-entropy.min())
            confidence = 1-entropy
            
            # plt.hist(confidence.reshape(-1))
            # plt.show()
            # import pdb;pdb.set_trace()
            # confidence = (1-entropy)/(1-entropy).max()
            
            fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
            fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

            if padding:
                fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
                fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
                confidence = np.pad(confidence, ((1, 1), (1, 1)), 'constant')
                gt_pad = np.pad(self.gt_mask, ((1, 1), (1, 1)), 'constant')
                pred_mask_pad = np.pad(pred_mask, ((1, 1), (1, 1)), 'constant')
                # fn_mask = fn_mask * (confidence<0.6)
                # fp_mask = fp_mask * (confidence<0.6)
                # if len(self.clicks_list)<=5:
                #     fn_mask = fn_mask * (confidence>0.6)
                #     fp_mask = fp_mask * (confidence>0.6)
                # else:
                #     fn_mask = fn_mask * (confidence<0.6)
                #     fp_mask = fp_mask * (confidence<0.6)
            
            fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
            fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
            
            
            
            # th = 0.8
            high_conf = confidence > 0.8
            low_conf = confidence < 0.2
            
            high_labels_map, high_num = label(high_conf)
            low_labels_map, low_num = label(low_conf)
            
            high_conf_top, high_conf_count =  self.get_top_labels(high_labels_map, 10)
            low_conf_top, low_conf_count = self.get_top_labels(low_labels_map, 10)
            # import pdb;pdb.set_trace()
            
            # os.makedirs('./experiments/evaluation_logs/camo/sam_b_mask1_conf_fixed_0.8_case3user/vis/', exist_ok=True)
            # # plt.imsave('./analysis/fn_mask_unconfand'+str(len(self.clicks_list))+'.png', fn_mask*(confidence<0.5))
            # # plt.imsave('./analysis/fp_mask_unconfand'+str(len(self.clicks_list))+'.png', fp_mask*(confidence<0.5))
            # # plt.imsave('./analysis/fn_mask_unconfor'+str(len(self.clicks_list))+'.png', fn_mask+gt_pad*(confidence<0.5))
            # # plt.imsave('./analysis/fp_mask_unconfor'+str(len(self.clicks_list))+'.png', ~fp_mask+gt_pad*(confidence<0.5))
            # plt.imsave('./experiments/evaluation_logs/camo/sam_b_mask1_conf_fixed_0.8_case3user/vis/'+sample_id.rstrip('.jpg')+'confidence'+str(len(self.clicks_list))+'.png', confidence)
            # plt.imsave('./experiments/evaluation_logs/camo/sam_b_mask1_conf_fixed_0.8_case3user/vis/'+sample_id.rstrip('.jpg')+'pred'+str(len(self.clicks_list))+'.png', pred_mask)
            # plt.imsave('./experiments/evaluation_logs/camo/sam_b_mask1_conf_fixed_0.8_case3user/vis/'+sample_id.rstrip('.jpg')+'high_conf'+str(len(self.clicks_list))+'.png', high_labels_map*np.isin(high_labels_map, high_conf_top))
            # plt.imsave('./experiments/evaluation_logs/camo/sam_b_mask1_conf_fixed_0.8_case3user/vis/'+sample_id.rstrip('.jpg')+'low_conf'+str(len(self.clicks_list))+'.png', low_labels_map*np.isin(low_labels_map, low_conf_top))

            click_coords = []
            for prev_click in self.clicks_list:
                click_coords.append(prev_click.coords)
                
            click_coords = np.array(click_coords)
            
            if high_num == 0:
                import pdb;pdb.set_trace()
            high_conf_points = []
            for high_label in high_conf_top:
                high_label_mask = np.isin(high_labels_map, high_label)
                
                if high_label_mask.sum() < (high_labels_map.shape[0]*high_labels_map.shape[1])/500:
                    break
                # high_label_mask = np.pad(high_label_mask, ((1, 1), (1, 1)), 'constant')
                high_label_mask_dt = cv2.distanceTransform(high_label_mask.astype(np.uint8), cv2.DIST_L2, 0)
                high_label_mask_dt = high_label_mask_dt[1:-1, 1:-1]
                conf_point_y, conf_point_x = np.where(high_label_mask_dt == high_label_mask_dt.max())
                high_conf_points.append([conf_point_y[0], conf_point_x[0]])
            
            high_conf_points =np.array(high_conf_points)
            
            if high_conf_points.shape[0] ==0:
                high_conf_sorted_points =np.array([])
            else:
                high_conf_sorted_points, high_conf_dist = self.find_farthest_point_vectorized(click_coords, high_conf_points)
            # high_conf_sorted_points, high_conf_dist = self.find_farthest_point_vectorized(click_coords, high_conf_points)
            
            for high_conf_point in high_conf_sorted_points:
                # import pdb;pdb.set_trace()
                if pred_mask[high_conf_point[0], high_conf_point[1]] != self.gt_mask[high_conf_point[0], high_conf_point[1]]:
                    print("case 1")
                    return Click(is_positive=self.gt_mask[high_conf_point[0], high_conf_point[1]], coords=(high_conf_point[0], high_conf_point[1]))
            
            low_conf_points = []
            for low_label in low_conf_top:
                low_label_mask = np.isin(low_labels_map, low_label)
                
                if low_label_mask.sum() < (low_labels_map.shape[0]*low_labels_map.shape[1])/500:
                    break
                # high_label_mask = np.pad(high_label_mask, ((1, 1), (1, 1)), 'constant')
                low_label_mask_dt = cv2.distanceTransform(low_label_mask.astype(np.uint8), cv2.DIST_L2, 0)
                low_label_mask_dt = low_label_mask_dt[1:-1, 1:-1]
                conf_point_y, conf_point_x = np.where(low_label_mask_dt == low_label_mask_dt.max())
                low_conf_points.append([conf_point_y[0], conf_point_x[0]])
            
            low_conf_points =np.array(low_conf_points)
            
            if low_conf_points.shape[0] ==0:
                low_conf_sorted_points =np.array([])
            else:
                low_conf_sorted_points, low_conf_dist = self.find_farthest_point_vectorized(click_coords, low_conf_points)
            
            # low_conf_sorted_points, low_conf_dist = self.find_farthest_point_vectorized(click_coords, low_conf_points)
            
            for low_conf_point in low_conf_sorted_points:
                if pred_mask[low_conf_point[0], low_conf_point[1]] != self.gt_mask[low_conf_point[0], low_conf_point[1]]:
                    print("case 2")
                    return Click(is_positive=self.gt_mask[low_conf_point[0], low_conf_point[1]], coords=(low_conf_point[0], low_conf_point[1]))
            
            # print("case 3")
            # return Click(is_positive=self.gt_mask[low_conf_sorted_points[0][0], low_conf_sorted_points[0][1]], coords=(low_conf_sorted_points[0][0], low_conf_sorted_points[0][1]))
            
            
            
            # import pdb;pdb.set_trace()
            # for high_label in high_conf_top:
            #     high_label_mask = np.isin(high_labels_map, high_label)
            #     # high_label_mask = np.pad(high_label_mask, ((1, 1), (1, 1)), 'constant')
            #     high_label_mask_dt = cv2.distanceTransform(high_label_mask.astype(np.uint8), cv2.DIST_L2, 0)
            #     high_label_mask_dt = high_label_mask_dt[1:-1, 1:-1]
            #     conf_point_y, conf_point_x = np.where(high_label_mask_dt == high_label_mask_dt.max())
                
            #     if pred_mask[conf_point_y[0], conf_point_x[0]] != self.gt_mask[conf_point_y[0], conf_point_x[0]]:
            #         # print("case 1")
            #         return Click(is_positive=self.gt_mask[conf_point_y[0], conf_point_x[0]], coords=(conf_point_y[0], conf_point_x[0]))
            
            # for low_label in low_conf_top:
            #     low_label_mask = np.isin(low_labels_map, low_label)
            #     low_label_mask_dt = cv2.distanceTransform(low_label_mask.astype(np.uint8), cv2.DIST_L2, 0)
            #     low_label_mask_dt= low_label_mask_dt[1:-1, 1:-1]
            #     conf_point_y, conf_point_x = np.where(low_label_mask_dt == low_label_mask_dt.max())
                
            #     if pred_mask[conf_point_y[0], conf_point_x[0]] != self.gt_mask[conf_point_y[0], conf_point_x[0]]:
            #         # print("case 2")
            #         return Click(is_positive=self.gt_mask[conf_point_y[0], conf_point_x[0]], coords=(conf_point_y[0], conf_point_x[0]))
            
            # low_label_mask = np.isin(low_labels_map, low_conf_top[0])
            # low_label_mask_dt = cv2.distanceTransform(low_label_mask.astype(np.uint8), cv2.DIST_L2, 0)
            # low_label_mask_dt= low_label_mask_dt[1:-1, 1:-1]
            # conf_point_y, conf_point_x = np.where(low_label_mask_dt == low_label_mask_dt.max())
            # return Click(is_positive=self.gt_mask[conf_point_y[0], conf_point_x[0]], coords=(conf_point_y[0], conf_point_x[0]))
        
            
            
            # high_conf_top_mask = np.isin(high_labels, high_conf_top)
            # low_conf_top_mask = np.isin(low_labels, low_conf_top)
            
            # fn_mask_dt = self.compute_score(fn_mask_dt, confidence)
            # fp_mask_dt = self.compute_score(fp_mask_dt, confidence)
            
            
            # import pdb;pdb.set_trace()
            if padding:
                fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
                fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

            fn_mask_dt = fn_mask_dt * self.not_clicked_map
            fp_mask_dt = fp_mask_dt * self.not_clicked_map

            fn_max_dist = np.max(fn_mask_dt)
            fp_max_dist = np.max(fp_mask_dt)

            is_positive = fn_max_dist > fp_max_dist
            if is_positive:
                coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
            else:
                coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
            
            print("case 3")
            return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)


        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def get_clicks_map(self):
        map_shape = self.gt_mask.shape
        clicks_map = np.full(map_shape, -1, dtype=np.int32)
        
        for click in self.clicks_list:
            if click.is_positive:
                clicks_map[click.coords[0], click.coords[1]] = 1
            else:
                clicks_map[click.coords[0], click.coords[1]] = 0
        
        return clicks_map

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool_)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
