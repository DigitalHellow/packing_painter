"""
Example showing how to separate the image into
sections. Each section has its own independent
implementation
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt

from optimizer import utils
from optimizer import packing_optimizer as po

target_image = "imgs/Reserva_preto.jpg"
image = utils.load_bin_image(target_image, True)


example = ["circle", "rect", "square"] # rect, square or circle

def option_to_array(option: str):
    if option == "rect":
        R = np.array([(8, 16), (24, 32), (48, 64), (128,128)]) # rect sizes
    elif option == "circle":
        R = np.array([8, 16, 24, 32, 48, 64]) # circle radius
    elif option == "square":
        R = np.array([(8,8)]) # square
    else:
        print("Invalid option. Using circle option")
        R = np.array([8, 16, 24, 32, 48, 64]) # circle radius
    return R

Rs = [option_to_array(e) for e in example]

n = 3
opts = po.SecPackingOptimizer(n, Rs, image)
centers = opts(5)

c = np.array(
    [opt.calc_mask(*center) for opt in opts.opts
        for center in opt.centers]
).sum(0)

plt.figure(figsize=(15, 15))
plt.imshow(image - c, cmap="gray")

data = opts.save_solution("data_seg", 
       [[f"imgs/Cards/{f}" for f in os.listdir("imgs/Cards/")]] * n,
    scale=4)
# %%
