import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F


class FacialLMBasicBlock(nn.Module):
    """ Building block for mediapipe facial landmark model

    DepthwiseConv + Conv + PRelu
    downsampling + channel padding for few blocks(when stride=2)
    channel padding values - 16, 32, 64

    Args:
        nn ([type]): [description]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(FacialLMBasicBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.depthwiseconv_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.prelu = nn.PReLU(out_channels)


    def forward(self, x):
        """[summary]

        Args:
            x ([torch.Tensor]): [input tensor]

        Returns:
            [torch.Tensor]: [featues]
        """

        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.prelu(self.depthwiseconv_conv(h) + x)


def pad_image(im, desired_size=192):
    """[summary]

    Args:
        im ([cv2 image]): [input image]
        desired_size (int, optional): [description]. Defaults to 64.

    Returns:
        [cv2 image]: [resized image]
    """
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im


class GetKeysDict:
    """
    maps pytorch keys to tflite keys
    """
    def __init__(self):
        self.facial_landmark_dict = {
                                        'confidence.2.depthwiseconv_conv.0.weight': 'depthwise_conv2d_16/Kernel',
                                        'confidence.2.depthwiseconv_conv.0.bias' : 'depthwise_conv2d_16/Bias',
                                        'confidence.2.depthwiseconv_conv.1.weight': 'conv2d_17/Kernel',
                                        'confidence.2.depthwiseconv_conv.1.bias': 'conv2d_17/Bias',
                                        'confidence.2.prelu.weight': 'p_re_lu_17/Alpha',

                                        'confidence.3.weight': 'conv2d_18/Kernel',
                                        'confidence.3.bias': 'conv2d_18/Bias',
                                        'confidence.4.weight': 'p_re_lu_18/Alpha',

                                        'confidence.5.depthwiseconv_conv.0.weight': 'depthwise_conv2d_17/Kernel',
                                        'confidence.5.depthwiseconv_conv.0.bias' : 'depthwise_conv2d_17/Bias',
                                        'confidence.5.depthwiseconv_conv.1.weight': 'conv2d_19/Kernel',
                                        'confidence.5.depthwiseconv_conv.1.bias': 'conv2d_19/Bias',
                                        'confidence.5.prelu.weight': 'p_re_lu_19/Alpha',

                                    

                                        'confidence.6.weight': 'conv2d_20/Kernel',
                                        'confidence.6.bias': 'conv2d_20/Bias',

                                        'facial_landmarks.0.depthwiseconv_conv.0.weight': 'depthwise_conv2d_22/Kernel',
                                        'facial_landmarks.0.depthwiseconv_conv.0.bias': 'depthwise_conv2d_22/Bias',
                                        'facial_landmarks.0.depthwiseconv_conv.1.weight': 'conv2d_27/Kernel',
                                        'facial_landmarks.0.depthwiseconv_conv.1.bias': 'conv2d_27/Bias',
                                        'facial_landmarks.0.prelu.weight': 'p_re_lu_25/Alpha',


                                        'facial_landmarks.1.weight': 'conv2d_28/Kernel',
                                        'facial_landmarks.1.bias': 'conv2d_28/Bias',
                                        'facial_landmarks.2.weight': 'p_re_lu_26/Alpha',
                                        
                                        'facial_landmarks.3.depthwiseconv_conv.0.weight': 'depthwise_conv2d_23/Kernel',
                                        'facial_landmarks.3.depthwiseconv_conv.0.bias': 'depthwise_conv2d_23/Bias',
                                        'facial_landmarks.3.depthwiseconv_conv.1.weight': 'conv2d_29/Kernel',
                                        'facial_landmarks.3.depthwiseconv_conv.1.bias': 'conv2d_29/Bias',
                                        'facial_landmarks.3.prelu.weight': 'p_re_lu_27/Alpha',
                                        
                                        'facial_landmarks.4.weight': 'conv2d_30/Kernel',
                                        'facial_landmarks.4.bias': 'conv2d_30/Bias',

                                    }