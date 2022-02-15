import os
import numpy as np
import matplotlib.pyplot as plt

from optimizer import utils
from optimizer import packing_optimizer as po
from optimizer import object_segmentation as ob_seg

target_image = "imgs/Reserva_preto.jpg"
image = utils.load_bin_image(target_image, True)


example = "rect" # rect, square or circle

if example == "rect":
    R = np.array([(8, 16), (24, 32), (48, 64), (128,128),
        (64, 48), (16, 8), (32, 24)]) # rect sizes
elif example == "circle":
    R = np.array([8, 16, 24, 32, 48, 64]) # circle radius
elif example == "square":
    R = np.array([(8,8)]) # square
else:
    print("Invalid option. Using circle option")
    R = np.array([8, 16, 24, 32, 48, 64]) # circle radius


opt = po.PackingOptimizer(R, image)
centers = opt(10, 0.1, True)

c = np.array(
    [opt.calc_mask(*p) for p in centers]
).sum(0)

# save data to use with open gl
# opt.save_solution("data", ["imgs/masked_0.png", "imgs/masked_1.png",
#     "imgs/masked_2.png"])

# show solution
plt.figure(figsize=(15, 15))
plt.imshow(image - c, cmap="gray")
plt.title("Best solution")
plt.show()


opt.save_solution("data", 
        [f"imgs/Cards/{f}" for f in os.listdir("imgs/Cards/")],
    scale=4)