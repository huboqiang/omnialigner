import random
from PIL import Image
import numpy as np
import cv2

from torchvision import datasets, transforms
from PIL import ImageFilter, ImageOps

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def guided_filter(I, p, r, eps):
    """Guided Filter.
    
    Parameters:
    I -- guidance image (should be a gray-scale/single channel image)
    p -- filtering input image (should be a gray-scale/single channel image)
    r -- the radius of the window (mean filter size will be (2r+1)x(2r+1))
    eps -- regularization parameter
    
    Returns:
    q -- filtered output image
    """
    if type(I) == Image.Image:
        I = np.array(I)
    
    if type(p) == Image.Image:
        p = np.array(p)
    
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    
    q = mean_a * I + mean_b
    return q

class EdgeTransform(object):
    """
    Apply Sobel edge detection and bilateral filtering to the PIL image with a given probability.
    """
    def __init__(self, p=0.5, radius=10, filter_type="bilateral"):
        self.prob = p
        self.radius = radius
        self.filter_type = filter_type

    def get_image(self, img, img_p=None):
        # Convert the image to a NumPy array
        image = np.array(img)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Enhance the left and right halves of the image

        gray_enhanced = cv2.equalizeHist(gray_image)
        # right_gray_enhanced = cv2.equalizeHist(right_gray)

        # Apply bilateral filter to the left and right enhanced grayscale images
        if self.filter_type == "bilateral":
            gray_enhanced_filtered = cv2.bilateralFilter(gray_enhanced, d=self.radius, sigmaColor=75, sigmaSpace=75)
        elif self.filter_type == "guided":
            gray_enhanced_p = gray_enhanced
            if img_p is not None:
                gray_image_p = cv2.cvtColor(np.array(img_p), cv2.COLOR_RGB2GRAY)
                gray_enhanced_p = cv2.equalizeHist(gray_image_p)
            
            gray_enhanced_filtered = guided_filter( gray_enhanced, gray_enhanced_p, r=self.radius, eps=0.01)

        # Apply Sobel edge detection on the filtered left half
        sobelx_enhanced_filtered = cv2.Sobel(gray_enhanced_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobely_enhanced_filtered = cv2.Sobel(gray_enhanced_filtered, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges_enhanced_filtered = cv2.magnitude(sobelx_enhanced_filtered, sobely_enhanced_filtered)
        sobel_edges_enhanced_filtered = cv2.normalize(sobel_edges_enhanced_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create a new image with the Sobel edges on both halves
        output_combined_enhanced_filtered = cv2.cvtColor(sobel_edges_enhanced_filtered, cv2.COLOR_GRAY2RGB)
        return output_combined_enhanced_filtered
    
    def __call__(self, img, img_p=None):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        output_combined_enhanced_filtered = self.get_image(img, img_p)
        return Image.fromarray(output_combined_enhanced_filtered)



class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # EdgeTransformSHG(0.5),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
