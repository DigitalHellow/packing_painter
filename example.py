"""
Running example of how to use the packing optimizer
"""

# %%
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

from optimizer import utils
from optimizer import packing_optimizer as po
from optimizer import object_segmentation as ob_seg

target_image = "imgs/Reserva_preto.jpg"
image = utils.load_bin_image(target_image, True)
flowers_path = ["imgs/flower.jpeg", "imgs/flower_pink.jpeg",
    "imgs/flower2.jpeg"]

# %%
# optimize
R = np.array([8, 16, 24, 32, 48, 64]) # circle radius

opt = po.PackingOptimizer(R, image)
centers = opt()

c = np.array(
    [opt.circle_mask(*p) for p in centers]
).sum(0)

# show solution
plt.imshow(image - c, cmap="gray")

# %%
# get flowers from our image
# parameters from segmentation
# where chosen after a few tries
edges_params = [(30, 150), (30, 150), (30, 200)]
grays = [0, 1, 1]
blurs = [0, 0, 1]
filter_idx = [1, 2, 1]


flowers = []
for i, flower_path in enumerate(flowers_path):
    gray = grays[i]
    blur = blurs[i]
    edge_params = edges_params[i]

    _flower = cv2.imread(flower_path)
    if blur:
        _flower = cv2.GaussianBlur(_flower, (5,5), 0)

    flower = cv2.cvtColor(_flower, cv2.COLOR_BGR2RGB)
    seg_img = cv2.cvtColor(_flower, 
        cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB)
    seg, labeled_img = ob_seg.get_segmentation(seg_img, *edge_params, True)
    flowers.append((flower, labeled_img))

    # show image and extracted labels
    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(flower)
    axarr[1].imshow(labeled_img)
    plt.tight_layout()
    plt.show()

labeled_filters = [ob_seg.filter_label(flower, labeled_img, filter_idx[i]) 
    for i, (flower, labeled_img) in enumerate(flowers)]

# check if we filtered the images
f, axarr = plt.subplots(1,len(labeled_filters)) 
for i, labeled_filter in enumerate(labeled_filters):
    axarr[i].imshow(labeled_filter)
plt.tight_layout()
plt.show()

# %%
# resize image
flower_images = {}
scale = 1.5

for r in R:
    h = int(2*scale*r)
    w = int(2*scale*r)
    flower_images[r] = [
        cv2.resize(labeled_filter, (w+1, h+1))
        for labeled_filter in labeled_filters
    ]

# create new image using the flowers
should_rotate = True
synthetic_img = np.zeros((*image.shape, 4), dtype=np.uint8)
# add the biggest flowers last, so that they stay on top
# of the generated image
centers.sort(key=lambda c: c[-1])
for center in centers:
    x, y, r = center
    angle = np.random.randint(360) if should_rotate else 0
    flower_img = random.choice(flower_images[r])
    flower_img = rotate(flower_img, angle=angle, reshape=False)
    mask = utils.rec_mask(image, x, y, int(scale*r)+1)

    overlayed = utils.overlay(synthetic_img[mask], flower_img)
    synthetic_img[mask] = overlayed.reshape(-1,4)

plt.imshow(synthetic_img)

# save output
cv2.imwrite("imgs/out.png", synthetic_img)
# %%
