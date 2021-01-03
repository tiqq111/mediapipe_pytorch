import cv2
import tensorflow as tf
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facial_lm_model import FacialLM_Model
from utils import pad_image


class FaceMesh:
    """
    mediapipe face mesh model inefernce in pytorch and tflite
    """
    def __init__(self, model_path=None):
        """[summary]

        Args:
            model_path ([type], optional): [description]. Defaults to None.
        """
        # @TODO change model_path
        # tflite model
        self.interpreter = tf.lite.Interpreter("model_weights/face_landmark.tflite") 

        # pytorch model
        self.torch_model = FacialLM_Model()
        weights = torch.load('model_weights/facial_landmarks.pth')
        self.torch_model.load_state_dict(weights)
        self.torch_model = self.torch_model.eval()


    def __call__(self, img_path):
        """[summary]

        Args:
            img_path ([str]): [image path]

        Returns:
            [list]: [face landmarks and confidence]
        """
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()
        blob = cv2.imread(img_path).astype(np.float32)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        
        blob = pad_image(blob, desired_size=192)
        
        # -1 to 1 norm
        # blob /= 255 # x.float() / 127.5 - 1.0
        # @TODO /255 works better for few images
        blob = (blob / 127.5) - 1.0
        # blob = (blob - 0.5) / 0.5
        # blob = blob / 127.5 
        # blob = (blob - 128) / 255.0
        
        facial_landmarks_torch, confidence_torch = self.torch_model.predict(blob)

        blob = np.expand_dims(blob, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], blob)

        self.interpreter.invoke()

        facial_landmarks = self.interpreter.get_tensor(self.output_details[0]['index'])
        # confidence   = self.interpreter.get_tensor(self.output_details[1]['index'])
        
        # np.testing.assert_array_almost_equal(facial_landmarks_torch.cpu().detach().numpy(), facial_landmarks, decimal=3)
        print("Tensorrt and torch values are matching ::", np.allclose(facial_landmarks_torch.cpu().detach().numpy(), facial_landmarks, atol=1e-02))
        return facial_landmarks_torch, confidence_torch


m = FaceMesh()
img_path = '4.jpg'
facial_landmarks_torch, confidence_torch = m(img_path)

im = cv2.imread(img_path)
im = pad_image(im, desired_size=192)

facial_landmarks_ = facial_landmarks_torch.reshape(-1)
np.save('output', facial_landmarks_)
for idx in range(468):
    cv2.circle(im, (int(facial_landmarks_[idx*3]), int(facial_landmarks_[idx*3 + 1])), 1, (200, 160, 75), -1)

# cv2.imwrite('output.jpg', im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(im)
plt.show()