'''
@Time    : 2022/5/6 16:08
@Author  : leeguandon@gmail.com
'''
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmocr.datasets import IcdarDataset


@DATASETS.register_module()
class IcdarDatasetAdd(IcdarDataset):
    """Dataset for text detection while ann_file in coco format.

    Args:
        ann_file_backend (str): Storage backend for annotation file,
            should be one in ['disk', 'petrel', 'http']. Default to 'disk'.
    """
    CLASSES = ('text')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 select_first_k=-1,
                 ann_file_backend='disk'):
        super(IcdarDataset, self).__init__(ann_file, pipeline, classes, data_root, img_prefix,
                                           seg_prefix, proposal_file, test_mode, filter_empty_gt, select_first_k, ann_file_backend)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        try:
            if self.test_mode:
                return self.prepare_test_img(idx)
            while True:
                data = self.prepare_train_img(idx)
                if data is None:
                    idx = self._rand_another(idx)
                    continue
                return data
        except:
            return self.__getitem__(np.random.randint(self.__len__()))
