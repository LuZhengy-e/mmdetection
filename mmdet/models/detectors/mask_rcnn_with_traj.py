from ..builder import DETECTORS
from .two_backbones import TwoBackboneDetector


@DETECTORS.register_module()
class MaskRTCNN(TwoBackboneDetector):
    def __init__(self, 
                 backbone1, 
                 backbone2, 
                 neck1, neck2, 
                 rpn_head, 
                 roi_head, 
                 train_cfg, 
                 test_cfg, 
                 pretrained=None, 
                 init_cfg=None):
        super(MaskRTCNN, self).__init__(backbone1, 
                                        backbone2, 
                                        neck1, 
                                        neck2, 
                                        rpn_head, 
                                        roi_head, 
                                        train_cfg, 
                                        test_cfg, 
                                        pretrained, 
                                        init_cfg)