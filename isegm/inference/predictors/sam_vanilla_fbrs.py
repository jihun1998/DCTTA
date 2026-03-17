import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide, ResizeTrans
from isegm.utils.crop_local import  map_point_in_bbox,get_focus_cropv1, get_focus_cropv2, get_object_crop, get_click_crop
import numpy as np
import pdb
import torch.nn as nn

class SamPredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 infer_size = 256,
                 focus_crop_r = 1.4,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = []
        self.crop_l = infer_size
        self.focus_crop_r = focus_crop_r
        self.focus_roi = None 
        self.global_roi = None 
        self.prev_logit = None
        # self.scale = torch.ones(256, device='cuda', requires_grad=True)
        # self.bias = torch.zeros(256, device='cuda', requires_grad=True)
        self.scale = nn.Parameter(torch.ones(256, device='cuda'))  # 1.0으로 초기화
        self.bias = nn.Parameter(torch.zeros(256, device='cuda'))  # 0.0으로 초기화
        
    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        
        # Initialize
        img_np = self.original_image.squeeze().permute(1,2,0).cpu().detach().numpy()
        self.net.set_image(img_np)

    def set_scale_bias(self):
        # self.scale = torch.ones(256, device='cuda', requires_grad=True)
        # self.bias = torch.zeros(256, device='cuda', requires_grad=True)
        # self.scale = nn.Parameter(torch.empty(256).uniform_(0.999, 1.001).to('cuda'))
        # self.bias = nn.Parameter(torch.empty(256).uniform_(-0.001, 0.001).to('cuda'))
        self.scale = nn.Parameter(torch.ones(256, device='cuda'))  # 1.0으로 초기화
        self.bias = nn.Parameter(torch.zeros(256, device='cuda'))  # 0.0으로 초기화

    def set_prev_mask(self, mask):
        self.prev_prediction = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()

    def get_prediction(self, clicker, prev_mask=None,  mode='eval'):
        clicks_list = clicker.get_clicks()
        click = clicks_list[-1]
        last_y,last_x = click.coords[0],click.coords[1]
        self.last_y = last_y
        self.last_x = last_x

        img = self.original_image
        mask, logit, logit_256 = self._get_prediction(img, [clicks_list], prev_mask=prev_mask)
        
        return mask, logit, logit_256

    def _get_prediction(self, img, clicks_lists, prev_mask=None):
        points_nd, click_label = self.get_points_nd(clicks_lists)
        points_nd[:, [0, 1]] = points_nd[:, [1, 0]]
        
        
        # import pdb;pdb.set_trace()
        s = self.scale.view(1,-1,1,1)
        b = self.bias.view(1,-1,1,1)
        # brs_feature=self.net.features * s + b
        # self.net.features =  brs_feature

        flag_multimask = False
        idx_mask = 0
        if prev_mask is None:
            flag_multimask = True
            idx_mask = 1

        sam_output = self.net.predict_with_gradient_brs(point_coords=points_nd.cpu().detach().numpy(),
                                                    point_labels=click_label.cpu().detach().numpy(),
                                                    mask_input=prev_mask,
                                                    multimask_output=flag_multimask, 
                                                    return_logits=True,
                                                    scale=s,
                                                    bias=b
                                                    )

        logits, scores, logits_256 = sam_output
        # import pdb;pdb.set_trace()
        masks = logits>0
        return masks[idx_mask], logits[idx_mask], logits_256[idx_mask]

    def get_prediction2(self, clicker, prev_mask=None,  mode='eval'):
        clicks_list = clicker.get_clicks()
        click = clicks_list[-1]
        last_y,last_x = click.coords[0],click.coords[1]
        self.last_y = last_y
        self.last_x = last_x

        img = self.original_image
        mask, logit, logit_256 = self._get_prediction(img, [clicks_list], prev_mask=prev_mask)
        
        return mask, logit, logit_256

    def _get_prediction2(self, img, clicks_lists, prev_mask=None):
        points_nd, click_label = self.get_points_nd(clicks_lists)
        points_nd[:, [0, 1]] = points_nd[:, [1, 0]]
        
        # Initialize
        img_np = img.squeeze().permute(1,2,0).cpu().detach().numpy()
        self.net.set_image(img_np)

        flag_multimask = False
        idx_mask = 0
        if prev_mask is None:
            flag_multimask = True
            idx_mask = 0

        sam_output = self.net.predict_with_gradient(point_coords=points_nd.cpu().detach().numpy(),
                                                    point_labels=click_label.cpu().detach().numpy(),
                                                    mask_input=prev_mask,
                                                    multimask_output=flag_multimask, 
                                                    return_logits=True)

        logits, scores, logits_256 = sam_output
        # import pdb;pdb.set_trace()
        masks = logits>0
        return masks[idx_mask], logits[idx_mask], logits_256[idx_mask]

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)
        print('_set_transform_states')

    def get_points_nd(self, clicks_lists):
        # import pdb;pdb.set_trace()
        total_clicks = []
        total_label = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        # import pdb;pdb.set_trace()
        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            # pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
            total_label = total_label + [1]*len(pos_clicks)
            
            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            # neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)
            total_label = total_label + [0]*len(neg_clicks)
            
        return torch.tensor(total_clicks, device=self.device)[0][:,:2], torch.tensor(total_label, device=self.device)

    def get_points_nd_inbbox(self, clicks_list, y1,y2,x1,x2):
        total_clicks = []
        num_pos = sum(x.is_positive for x in clicks_list)
        num_neg =len(clicks_list) - num_pos 
        num_max_points = max(num_pos, num_neg)
        num_max_points = max(1, num_max_points)
        pos_clicks, neg_clicks = [],[]
        for click in clicks_list:
            flag,y,x,index = click.is_positive, click.coords[0],click.coords[1], 0
            y,x = map_point_in_bbox(y,x,y1,y2,x1,x2,self.crop_l)
            if flag:
                pos_clicks.append( (y,x,index))
            else:
                neg_clicks.append( (y,x,index) )

        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)


        

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        print('set')
