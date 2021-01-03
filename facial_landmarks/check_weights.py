import cv2
import tensorflow as tf
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facial_lm_model import FacialLM_Model
from tflite import Model

data = open("model_weights/face_landmark.tflite", "rb").read()
model = Model.GetRootAsModel(data, 0)

tflite_graph = model.Subgraphs(0)
tflite_graph.Name()

# Tensor name to index mapping
tflite_tensor_dict = {}
for i in range(tflite_graph.TensorsLength()):
    tflite_tensor_dict[tflite_graph.Tensors(i).Name().decode("utf8")] = i 


parameters = {}
for i in range(tflite_graph.TensorsLength()):
    tensor = tflite_graph.Tensors(i)
    if tensor.Buffer() > 0:
        name = tensor.Name().decode("utf8")
        parameters[name] = tensor.Buffer()
    else:
        # Buffer value less than zero are not weights
        print(tensor.Name().decode("utf8"))

print("Total parameters: ", len(parameters))


def get_weights(tensor_name):
    index = tflite_tensor_dict[tensor_name]
    tensor = tflite_graph.Tensors(index)

    buffer = tensor.Buffer()
    shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    weights = model.Buffers(buffer).DataAsNumpy()
    weights = weights.view(dtype=np.float32)
    weights = weights.reshape(shape)
    return weights

net = FacialLM_Model()
weights = torch.load('facial_landmarks.pth')
net.load_state_dict(weights)
net = net.eval()

# net(torch.randn(2,3,64,64))[0].shape

probable_names = []
for i in range(0, tflite_graph.TensorsLength()):
    tensor = tflite_graph.Tensors(i)
    if tensor.Buffer() > 0 and tensor.Type() == 0:
        probable_names.append(tensor.Name().decode("utf-8"))

pt2tflite_keys = {}
i = 0
for name, params in net.state_dict().items():
    print(name)
    if i < 83:
        pt2tflite_keys[name] = probable_names[i]
        i += 1

matched_keys = {
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

pt2tflite_keys.update(matched_keys)

for key, value in pt2tflite_keys.items():
    # print(key, value)
    tflite_ = parameters[value]
    W = get_weights(value)
    if W.ndim == 4:
        if 'depthwise' in value:
            # (1, 3, 3, 32) --> (32, 1, 3, 3)
            # for depthwise conv
            W = W.transpose((3, 0, 1, 2))  
        else:
            W = W.transpose((0, 3, 1, 2)) 
    elif W.ndim == 3:
        # prelu
        W = W.reshape(-1)
    tflite_ = W

    torch_ = net.state_dict()[key]
    # print(key, value, tflite_.shape, torch_.shape)
    np.testing.assert_array_almost_equal(torch_.cpu().detach().numpy(), tflite_, decimal=3)
    print("matching ::", np.allclose(torch_.cpu().detach().numpy(), tflite_, atol=1e-03))