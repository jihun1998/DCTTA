from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class TRASHCANDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='images', masks_dir_name='SegmentationGT/trash',
                 **kwargs):
        super(TRASHCANDataset, self).__init__(**kwargs)
        self.name = 'TRASHCAN'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        # self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self.dataset_samples = [x.name for x in sorted(self._insts_path.glob('*.*'))]

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name.replace('png','jpg'))
        mask_path = str(self._insts_path / image_name)
        # mask_path = str(self._masks_paths[image_name.split('.')[0]])

        # import pdb;pdb.set_trace()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        # instances_mask[instances_mask == 128] = -1
        # instances_mask[instances_mask > 128] = 1
        instances_mask = instances_mask > 128
        instances_mask = instances_mask.astype(np.int32)

        # return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)
        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
