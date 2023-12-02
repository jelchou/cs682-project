import cv2
import numpy as np
import random
import albumentations as A

def read_image(path):
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

def random_crop(x, target_r, target_c):
    r, c, *_ = x.shape
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(rand_r * (r - target_r)).astype(int)
    start_c = np.floor(rand_c * (c - target_c)).astype(int)
    return crop(x, start_r, start_c, target_r, target_c)

def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    r, c, *_ = im.shape
    M = cv2.getRotationMatrix2D((c / 2, r / 2), deg, 1)
    return cv2.warpAffine(im, M, (c, r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS + interpolation)

def normalize(im):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im / 255.0 - imagenet_stats[0]) / imagenet_stats[1]

def apply_transform_crop(x, sz=(224, 224), zoom=1.05):
    sz1 = int(zoom * sz[0])
    sz2 = int(zoom * sz[1])
    x = cv2.resize(x, (sz1, sz2))
    x = random_crop(x, sz[1], sz[0])
    return x

def apply_transform_rotation(x, sz=(224, 224), zoom=1.05):
    sz1 = int(zoom * sz[0])
    sz2 = int(zoom * sz[1])
    x = cv2.resize(x, (sz1, sz2))
    x = rotate_cv(x, np.random.uniform(-10, 10))
    return x

def apply_transform_rgb(x, sz=(224, 224)):
    transform = A.Compose([
        A.ToGray(),
        A.Resize(height=sz[0], width=sz[1])
    ])
    augmented = transform(image=x)
    x = augmented['image']
    return x
