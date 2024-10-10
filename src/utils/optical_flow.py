import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_optical_flow(previous_image, current_frame):
    """
    previous = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv = np.zeros_like(previous_image) # hue, saturation, value
    hsv[...,1] = 255 # saturation

    hsv[...,0] = angle*180/np.pi/2
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr
    """
    flow = current_frame - previous_image
    return flow

def get_optical_flow_batch(images):
    flows = []
    for i in range(len(images)-1):
        previous_image = images[i]
        current_image = images[i+1]
        flow = get_optical_flow(previous_image, current_image)
        flows.append(flow)
    return np.stack(flows)

def normalize(x):
    if x.max() == x.min():
        return x - x.min()
    return (x - x.min()) / (x.max() - x.min())

def normalized_optical_flow(previous_image, current_image):
    flow = get_optical_flow(np.moveaxis(previous_image, 0, -1), np.moveaxis(current_image, 0, -1))
    flow = normalize(flow)
    return np.moveaxis(flow, -1, 0)

def draw_optical_flow(previous_image, current_image):
    flow = get_optical_flow(np.moveaxis(previous_image, 0, -1), np.moveaxis(current_image, 0, -1))
    #flow = normalize(flow)
    flow = np.abs(flow)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].imshow(np.moveaxis(previous_image, 0, -1))
    axes[0].set_title('First frame')
    axes[0].axis('off')

    axes[1].imshow(np.moveaxis(current_image, 0, -1))
    axes[1].set_title('Second frame')
    axes[1].axis('off')

    axes[2].imshow(1-flow)
    axes[2].set_title('Optical flow')
    axes[2].axis('off')

    plt.show()