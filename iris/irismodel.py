import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import IrisBlock


class IrisLM(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self):
        """[summary]
        """
        super(IrisLM, self).__init__()

        self.backbone = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(64),

            IrisBlock(64, 64), IrisBlock(64, 64),
            IrisBlock(64, 64), IrisBlock(64, 64),
            IrisBlock(64, 128, stride=2),

            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2)
        )

        # iris_contour head
        self.iris_contour = nn.Sequential(
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=15, kernel_size=2, stride=1, padding=0, bias=True)
        )

        # eye_contour head
        self.eye_contour = nn.Sequential(
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=213, kernel_size=2, stride=1, padding=0, bias=True)
        )


    @torch.no_grad()
    def forward(self, x):
        """ forward prop

        Args:
            x ([torch.Tensor]): [input Tensor]

        Returns:
            [list]: [eye_contour, iris_contour]
            eye_contour (batch_size, 213)
            (71 points)
            (x, y, z)
            (x, y) corresponds to image pixel locations
            iris_contour (batch_size, 15)
            (5, 3) 5 points
        """
        with torch.no_grad():
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            # (_, 128, 8, 8)
            features = self.backbone(x)            

            # (_, 213, 1, 1)  
            eye_contour = self.eye_contour(features)            

            # (_, 15, 1, 1)
            iris_contour = self.iris_contour(features) 
        # (batch_size, 213)  (batch_size, 15)
        return [eye_contour.view(x.shape[0], -1), iris_contour.reshape(x.shape[0], -1)]


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

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        eye_contour, iris_contour = self.forward(x)

        return eye_contour.view(x.shape[0], -1), iris_contour.view(x.shape[0], -1)


    def test(self):
        """ Sample Inference"""
        inp = torch.randn(1, 3, 64, 64)
        output = self(inp)
        print(output[0].shape, output[1].shape)