import matplotlib.pyplot as plt
import numpy as np

############# OPTICAL FLOW #############
"""
    This module contains the functions used to compute the optical flow between two consecutive frames of a video sequence.
    The optical flow is a vector field that represents the motion of objects in the video sequence between two frames.

    The optical flow is computed comparing the intensity of pixels in the two frames, which estimates the motion of 
    objects in the video sequence.

    The optical flow is used in the GAN training process to generate realistic video sequences by predicting the motion
    of objects in the video frames.
"""

def get_optical_flow(previous_image, current_frame):
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