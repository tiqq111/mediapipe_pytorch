import cv2
import torch
import numpy as np
from facial_lm_model import FacialLM_Model
from utils import pad_image
import glob, os
from tqdm import tqdm


class FaceMesh:
    """
    mediapipe face mesh inefernce in pytorch
    """
    def __init__(self, model_path='model_weights/facial_landmarks.pth'):
        """[summary]

        Args:
            model_path (str, optional): [description]. 
                                        Defaults to 'model_weights/facial_landmarks.pth'.
        """
        self.torch_model = FacialLM_Model()
        weights = torch.load(model_path)
        self.torch_model.load_state_dict(weights)
        self.torch_model = self.torch_model.eval()


    def __call__(self, img_path):
        """[summary]

        Args:
            img_path ([str]): [image path]

        Returns:
            [list]: [face landmarks and confidence]
        """
        blob = cv2.imread(img_path).astype(np.float32)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        
        blob = pad_image(blob, desired_size=192)
        
        # -1 to 1 norm
        blob = (blob/127.5) - 1.0
        
        facial_landmarks_torch, confidence_torch = self.torch_model.predict(blob)

        return facial_landmarks_torch, confidence_torch


model = FaceMesh()
img_paths = glob.glob('test_images/*')
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

for img_path in tqdm(img_paths):
    facial_landmarks_torch, confidence_torch = model(img_path)

    im = cv2.imread(img_path)
    im = pad_image(im)

    facial_landmarks_ = facial_landmarks_torch.reshape(-1)
    
    for idx in range(468):
        cv2.circle(im, (int(facial_landmarks_[idx*3]), int(facial_landmarks_[idx*3 + 1])), 1, (0, 0, 255), -1)
        
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    filename = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(filename, im) 
    # import matplotlib.pyplot as plt
    # plt.imshow(im)
    # plt.show()