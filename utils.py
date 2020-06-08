"""
Utility functions

Author: Huan Minh Luu
Magnetic Resonance Imaging Laboratory
KAIST
luuminhhuan@kaist.ac.kr
"""
import numpy as np
from PIL import Image

def save_png(gt,prediction,save_dir):
    """
    Save a png of ground truth, prediction made by the network, and the difference 

    Parameters
    ----------
    gt : ground truth values
    
    prediction : output of the network

    save_dir : directory to save the image
    
    Returns
    -------
    None.
    The image is saved at save_dir

    """
    max_val = np.array([11.2,20.1])
    combined = np.zeros((256,384))
    
    gt_norm = gt/max_val
    prediction_norm = prediction/max_val
    diff = 5*np.abs(gt_norm - prediction_norm)
    for j in range(2):
        combined[128*j:128*(j+1),0:128] = gt_norm[:,:,j]
        combined[128*j:128*(j+1),128:256] = prediction_norm[:,:,j]
        combined[128*j:128*(j+1),256:] = diff[:,:,j]
    
    rescaled = (255*combined).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(save_dir)
    
                                                          
                                                          