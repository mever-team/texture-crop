import numpy as np
from PIL import Image
import random
from scipy import stats
from skimage import feature, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch
from torchvision.transforms import CenterCrop

############################################################### TEXTURE CROP ###############################################################

def texture_crop(image, stride=224, window_size=224, metric='he', position='top', n=10, drop = False):
    cropped_images = []
    images = []

    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))
    
    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray)
        if metric == 'sd':
            m = np.std(crop_gray / 255.0)
        elif metric == 'ghe':
            m = histogram_entropy_response(crop_gray / 255.0)
        elif metric == 'le':
            m = local_entropy_response(crop_gray)
        elif metric == 'ac':
            m = autocorrelation_response(crop_gray / 255.0)
        elif metric == 'td':
            m = texture_diversity_response(crop_gray / 255.0)
        images.append((crop, m))

    images.sort(key=lambda x: x[1], reverse=True)
    
    if position == 'top':
        texture_images = [img for img, _ in images[:n]]
    elif position == 'bottom':
        texture_images = [img for img, _ in images[-n:]]

    repeat_images = texture_images.copy()
    while len(texture_images) < n:
        texture_images.append(repeat_images[len(texture_images) % len(repeat_images)])

    return texture_images


def autocorrelation_response(image_array):
    """
    Calculates the average autocorrelation of the input image.
    """
    f = np.fft.fft2(image_array, norm='ortho')
    power_spectrum = np.abs(f) ** 2
    acf = np.fft.ifft2(power_spectrum, norm='ortho').real
    acf = np.fft.fftshift(acf)
    acf /= acf.max()
    acf = np.mean(acf)

    return acf

def histogram_entropy_response(image):
    """
    Calculates the entropy of the image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True) 
    prob_dist = histogram / histogram.sum()
    entr = stats.entropy(prob_dist + 1e-7, base=2)    # Adding a small value (1e-7) to avoid log(0)

    return entr

def local_entropy_response(image):
    """
    Calculates the spatial entropy of the image using a local entropy filter.
    """
    entropy_image = entropy(image, disk(10))  
    mean_entropy = np.mean(entropy_image)

    return mean_entropy

def texture_diversity_response(image):
    M = image.shape[0]  
    l_div = 0

    for i in range(M):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i, j + 1])

    # Vertical differences
    for i in range(M - 1):
        for j in range(M):
            l_div += abs(image[i, j] - image[i + 1, j])

    # Diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i + 1, j + 1])

    # Counter-diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i + 1, j] - image[i, j + 1])

    return l_div


############################################################## THRESHOLDTEXTURECROP ##############################################################

def threshold_texture_crop(image, stride=224, window_size=224, threshold=5, drop = False):
    cropped_images = []
    texture_images = []
    images = []

    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))

    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray) / 255.0
        
        histogram, _ = np.histogram(crop_gray.flatten(), bins=256, range=(0, 1), density=True) 
        prob_dist = histogram / histogram.sum()
        m = stats.entropy(prob_dist + 1e-7, base=2)
        if m > threshold: 
            texture_images.append(crop)

    if len(texture_images) == 0:
        texture_images = [CenterCrop(image)]

    return texture_images