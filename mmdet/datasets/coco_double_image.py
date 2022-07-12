import imp
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class CocoTwoImageDateset(CocoDataset):
    CLASSES = ("Road", "Intersection")

    def __init__(self, traj_prefix, **kwargs):
        super(CocoTwoImageDateset, self).__init__(**kwargs)
        self.traj_prefix = traj_prefix

    def pre_pipeline(self, results):
        results['traj_prefix'] = self.traj_prefix
        super(CocoTwoImageDateset, self).pre_pipeline(results)

