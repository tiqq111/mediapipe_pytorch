import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FacialLMBasicBlock


class FacialLM_Model(nn.Module):
    """[MediaPipe facial_landmark model in Pytorch]

    Args:
        nn ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 1x1x1x1404, 1x1x1x1
    # 1x1404x1x1

    def __init__(self):
        """[summary]"""
        super(FacialLM_Model, self).__init__()

        self.backbone = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(16),

            FacialLMBasicBlock(16, 16), 
            FacialLMBasicBlock(16, 16),
            FacialLMBasicBlock(16, 32, stride=2), # pad

            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 64, stride=2), 

            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 128, stride=2),

            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128, stride=2),

            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128)
        )

        # facial_landmark head
        # @TODO change name from self.confidence to self.facial_landmarks 
        self.confidence = nn.Sequential(
            FacialLMBasicBlock(128, 128, stride=2),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
            # ----
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(32),

            FacialLMBasicBlock(32, 32), 
            nn.Conv2d(in_channels=32, out_channels=1404, kernel_size=3, stride=3, padding=0, bias=True)
        )

        # confidence score head
        # @TODO change name from self.facial_landmarks to  self.confidence
        self.facial_landmarks = nn.Sequential(
            FacialLMBasicBlock(128, 128, stride=2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(32),

            FacialLMBasicBlock(32, 32), 
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=3, padding=0, bias=True)
        )


    @torch.no_grad()
    def forward(self, x):
        """ forward prop

        Args:
            x ([torch.Tensor]): [input Tensor]

        Returns:
            [list]: [facial_landmarks, confidence]
            facial_landmarks: 1 x 1404 x 1 x 1 
            (368 x 3)
            (x, y, z)
            (x, y) corresponds to image pixel locations
            confidence: 1 x 1 x 1 x 1
            368 face landmarks
        """
        
        # @TODO remove 
        with torch.no_grad():
            x = nn.ReflectionPad2d((1, 0, 1, 0))(x)
            features = self.backbone(x)            

            # @TODO change the names
            confidence = self.facial_landmarks(features)            

            facial_landmarks = self.confidence(features) 
        
            return [facial_landmarks, confidence]
        # return [facial_landmarks.view(x.shape[0], -1), confidence.reshape(x.shape[0], -1)]


    def predict(self, img):
        """ single image inference

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))
        
        return self.batch_predict(img.unsqueeze(0))


    def batch_predict(self, x):
        """ batch inference
        currently only single image inference is supported

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        facial_landmarks, confidence = self.forward(x)
        return facial_landmarks, confidence
        # return facial_landmarks.view(x.shape[0], -1), confidence.view(x.shape[0], -1)
    

    def test(self):
        """ Sample Inference"""
        inp = torch.randn(1, 3, 192, 192)
        output = self(inp)
        print(output[0].shape, output[1].shape)


# m = FacialLM_Model()
# m.test()
"""
m = FacialLM_Model()
inp = torch.randn(1, 3, 192, 192)
output = m(inp)
print(output[0].shape, output[1].shape)
"""