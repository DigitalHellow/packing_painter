"""
Running example of how to use the packing optimizer
"""

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from optimizer import utils
from optimizer import packing_optimizer as po
from optimizer import object_segmentation as ob_seg

target_image = "imgs/Reserva_preto.jpg"
image = utils.load_bin_image(target_image, True)
flowers_path = ["imgs/flower.jpeg", 
    "imgs/flower_pink.jpeg", "imgs/flower2.jpeg"]

# %%
# optimize
example = "rect" # rect, square or circle

if example == "rect":
    R = np.array([(8, 16), (24, 32), (48, 64), (128,128)]) # rect sizes
elif example == "circle":
    R = np.array([8, 16, 24, 32, 48, 64]) # circle radius
elif example == "square":
    R = np.array([(8,8)]) # square
else:
    print("Invalid option. Using circle option")
    R = np.array([8, 16, 24, 32, 48, 64]) # circle radius

opt = po.PackingOptimizer.from_images(flowers_path,
    [0.2, 0.1, 0.05], image)
#%%

opt = po.PackingOptimizer(R, image, True)
centers = opt(10, 0.2)

c = np.array(
    [opt.calc_mask(*p) for p in centers]
).sum(0)

# save data to use with open gl
# opt.save_solution("data", ["imgs/masked_0.png", "imgs/masked_1.png",
#     "imgs/masked_2.png"])

# show solution
plt.figure(figsize=(15, 15))
plt.imshow(image - c, cmap="gray")


opt.save_solution("data", 
        [f"imgs/Cards/{f}" for f in os.listdir("imgs/Cards/")],
    scale=4)
    
#%%

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))

c = np.array(
    [opt.calc_mask(*p) for p in opt.centers]
).sum(0)

# show solution
axes[0].imshow(image - c, cmap="gray")

c = np.array(
    [opt.calc_mask(*p) for p in opt.initial_solution]
).sum(0)

# show initial solution
axes[1].imshow(image - c, cmap="gray")

# %%
# Get flowers from our image.
# Parameters for segmentation where chosen after a few tries.
# If you have already masked your inputs you can just load it using opencv
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
    cv2.imwrite(f"imgs/masked_{i}.png", labeled_filter)
plt.tight_layout()
plt.show()

# %%
