import cv2
import tensorflow as tf
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from irismodel import IrisLM


def plot(img_path, iris, eye_contour):
    im = cv2.imread(img_path)
    im = pad_image(im, desired_size=64)

    lm = iris[0]
    h, w, _ = im.shape

    cv2.circle(im, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1)
    cv2.circle(im, (int(lm[3]), int(lm[4])), 1, (255, 0, 255), -1)
    cv2.circle(im, (int(lm[6]), int(lm[7])), 2, (255, 0, 255), -1)
    cv2.circle(im, (int(lm[9] ), int(lm[10])), 1, (255, 0, 255), -1)
    cv2.circle(im, (int(lm[12] ), int(lm[13])), 1, (255, 0, 255), -1)

    eye_contour
    for idx in range(71):
        cv2.circle(im, (int(eye_contour[0][idx*3]), int(eye_contour[0][idx*3 + 1])), 1, (0, 0, 255), -1)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    plt.show()


def pad_image(im, desired_size=64):
    
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

    new_im.shape
    return new_im


class Model:
    def __init__(self):
        """[summary]

        Args:
            model_path ([str]): [path]
        """
        # @TODO change path
        # add model paths for both pt and tflite models
        self.interpreter = tf.lite.Interpreter(model_path="model_weights/iris_landmark.tflite") # Model Loading 
        self.net = IrisLM()
        weights = torch.load('model_weights/irislandmarks.pth')
        self.net.load_state_dict(weights)
        self.net = self.net.eval()

    def __call__(self, img_path):
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(self.input_details)
        print(self.output_details)

        self.interpreter.allocate_tensors()
        blob = cv2.imread(img_path).astype(np.float32)
        blob = pad_image(blob, desired_size=64)
        
        blob /= 255 # x.float() / 127.5 - 1.0
        
        eye_contour_torch, iris_torch = self.net.predict(blob)

        blob = np.expand_dims(blob, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], blob)

        self.interpreter.invoke()

        eye_contour = self.interpreter.get_tensor(self.output_details[0]['index'])
        iris   = self.interpreter.get_tensor(self.output_details[1]['index'])
        np.testing.assert_array_almost_equal(eye_contour_torch.cpu().detach().numpy(), eye_contour, decimal=3)
        print("Are tflite and torch values matching? ::", np.allclose(eye_contour_torch.cpu().detach().numpy(), eye_contour, atol=1e-03))
        return eye_contour, iris

m = Model()
img_path = 'iris2.jpg'
eye_contour, iris = m(img_path)
plot(img_path, iris, eye_contour)