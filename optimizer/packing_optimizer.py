"""
Reconstructs an image using a given shape.
The reconstruction is done as if solving a packing
problem
"""

import cv2
import random
import numpy as np

from scipy.ndimage.interpolation import rotate

from typing import Iterable, List, Tuple

from . import utils

class PackingOptimizer:
    """
    Solves a packing problem to fill the target image
    with circles.
    """

    def __init__(self, R: Iterable[int], image: np.ndarray) -> None:
        self.centers: List[Tuple[int, int, int]]
        self.R = R.astype(np.int16) # circles radius
        
        self.image = image.copy() if len(image.shape) == 2 \
            else utils.load_bin_image(image, True)
        self.circle_mask = utils.mcircle_mask(self.image.shape)
        self._AREA = np.sum(self.image == 1)
        

    def __call__(self) -> List[Tuple[int, int, int]]:
        return self.optimize()


    def pick_center(self, picked_centers: List[Tuple[int, int, int]],
            image: np.ndarray, _R_copy = []) -> np.array:
        """
        Picks a random valid center from the image.
        picked_centers: array in the form (cx, cy, r). 
            Represents points already taken
        """
        if not _R_copy:
            R_copy = list(self.R)
            random.shuffle(R_copy)
        else:
            R_copy = _R_copy[:]

        while len(R_copy) > 0:
            r = R_copy.pop()
        
            circles = np.array(
                [self.circle_mask(p[0], p[1], p[2] + r) for p in picked_centers]
            ).sum(0)
            
            # possible locations to pick the center
            x, y = np.where(image - circles == 1)

            if (l:=len(x)) == 0:
                continue # if current radius doesn't fit go to next one

            idx = random.randrange(l)
            center = (y[idx], x[idx], r)

            return center
        
        return None # no radius fits the image


    def minimize_circle_loss(self, 
            center: Tuple[int, int, int], image: np.ndarray, 
            _picked_centers: List[Tuple[int, int, int]]) -> \
                Tuple[List[Tuple[int, int, int]], bool]:
        """
        Tries to minize the given circle loss by moving its
        center, lowering its radius and/or spliting it up 
        into more circles. Returns new centers and a boolean,
        which flags whether the new centers changed or not
        """
        # create a copy of the picked centers
        picked_centers = _picked_centers[:]
        picked_centers.remove(center)

        R_copy = list(self.R)
        # remove bigger radius
        R_copy = R_copy[:R_copy.index(center[2])]

        if len(R_copy) == 0:
            return (picked_centers, False)
        
        while (center:= self.pick_center(picked_centers, image, R_copy)) is not None:
            picked_centers.append(center)
        
        return (picked_centers, True)


    def circle_cost(self, center: Tuple[int, int, int],
            image: np.ndarray) -> int:
        """
        Circle loss is defined as the area outside the image
        """
        return self.area_outside(image, *center)


    def area_outside(self, img: np.ndarray, 
            cx: int, cy: int, r: int) -> int:
        """
        Returns the circle area outside the image
        """
        circle = self.circle_mask(cx, cy, r)
        return np.sum(circle | img) - self._AREA


    def total_cost(self, 
            picked_centers: List[Tuple[int, int, int]],
            image: np.ndarray) -> int:
        """
        Sum of circle's area outside the image and gaps 
        between circles
        """
        circles = np.array(
            [self.circle_mask(*p) for p in picked_centers]
        ).sum(0)
        cost = np.sum(image ^ circles)
        return cost

    
    def optimize(self) -> List[Tuple[int, int, int]]:
        """
        Solves the packing problem and returns the center coords and 
        radius of each circle
        """
        picked_centers = [] # initial solution
        while (center:= self.pick_center(picked_centers, self.image)) is not None:
            picked_centers += [center]

        # cost of each circle
        c_costs = [self.circle_cost(c, self.image) for c in picked_centers]

        # optimize circles which are outside the image
        n_picked_centers = picked_centers
        for i in range(50): # max of 50 rounds
            m = np.argmax(c_costs)
            _n_picked_centers, changed = self.minimize_circle_loss(
                n_picked_centers[m], self.image, n_picked_centers)
            
            if changed:
                n_picked_centers = _n_picked_centers
            else:
                break

            c_costs = [self.circle_cost(c, self.image) for c in n_picked_centers]
            if i%10 == 0:
                print(i)

        self.centers = n_picked_centers
        return n_picked_centers.copy()


    def make_image(self, scale: float,
            labeled_filters: np.ndarray) -> np.ndarray:
        flower_images = {}

        for r in self.R:
            h = int(2*scale*r)
            w = int(2*scale*r)
            flower_images[r] = [
                cv2.resize(labeled_filter, (w+1, h+1))
                for labeled_filter in labeled_filters
            ]

        # create new image using the flowers
        should_rotate = True
        synthetic_img = np.zeros((*self.image.shape, 4), dtype=np.uint8)
        # add the biggest flowers last, so that they stay on top
        # of the generated image
        self.centers.sort(key=lambda c: c[-1])
        for center in self.centers:
            x, y, r = center
            angle = np.random.randint(360) if should_rotate else 0
            flower_img = random.choice(flower_images[r])
            flower_img = rotate(flower_img, angle=angle, reshape=False)
            mask = utils.rec_mask(self.image, x, y, int(scale*r)+1)

            overlayed = utils.overlay(synthetic_img[mask], flower_img)
            synthetic_img[mask] = overlayed.reshape(-1,4)

        return synthetic_img