"""
Reconstructs an image using a given shape.
The reconstruction is done as if solving a packing
problem
"""

import cv2
import json
import random
import numpy as np

import scipy.signal as signal
from scipy.ndimage.interpolation import rotate

from typing import Iterable, List, Tuple, Union

from . import utils

# writing union all the time 
Size = Union[int, np.ndarray]

class PackingOptimizer:
    """
    Solves a packing problem to fill the target image
    with circles.
    """

    def __init__(self, R: Iterable[Size], image: np.ndarray) -> None:
        self.centers: List[Tuple[int, int, int]]
        self.image = image.copy() if len(image.shape) == 2 \
            else utils.load_bin_image(image, True)

        self.is_rec = hasattr(R[0], '__iter__')
        if self.is_rec:
            self.R = [np.array(r, dtype=np.int16) for r in R]
            # lambda just for the sweet lru cache
            self.calc_mask = lambda x, y, r: utils.mrec_mask(
                self.image.shape)(x, y, r[0], r[1])
        
        else:
            self.R = R.astype(np.int16) # circles radius
            self.calc_mask = utils.mcircle_mask(self.image.shape)

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
        
            masks = np.array(
                [self.calc_mask(p[0], p[1], p[2] + r) 
                for p in picked_centers]
            ).sum(0)
            
            # possible locations to pick the center
            x, y = np.where(image - masks == 1)

            if (l:=len(x)) == 0:
                continue # if current radius doesn't fit go to next one

            idx = random.randrange(l)
            center = (y[idx], x[idx], r)

            return center
        
        return None # no radius fits the image


    def minimize_circle_loss(self, 
            center: Tuple[int, int, int], image: np.ndarray, 
            _picked_centers: List[Tuple[int, int, Size]]) -> \
                Tuple[List[Tuple[int, int, Size]], bool]:
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
        if type(R_copy[0]) == np.ndarray:
            R_copy = R_copy[:next((i 
                for i, r in enumerate(R_copy) if r is center[2]))]    
        else:
            R_copy = R_copy[:R_copy.index(center[2])]

        if len(R_copy) == 0:
            return (picked_centers, False)
        
        while (center:= self.pick_center(picked_centers, 
                image, R_copy)) is not None:
            picked_centers.append(center)
        
        return (picked_centers, True)


    def out_cost(self, center: Tuple[int, int, Size],
            image: np.ndarray) -> int:
        """
        Area of the given object outside the image
        """
        return self.area_outside(image, *center)


    def area_outside(self, img: np.ndarray, cx: int, 
            cy: int, r: Size) -> int:
        """
        Returns the area outside the image
        """
        obj_area = self.calc_mask(cx, cy, r)
        return np.sum(obj_area | img) - self._AREA


    def total_cost(self, 
            picked_centers: List[Tuple[int, int, int]],
            image: np.ndarray) -> int:
        """
        Sum of circle's area outside the image and gaps 
        between circles
        """
        circles = np.array(
            [self.calc_mask(*p) for p in picked_centers]
        ).sum(0)
        cost = np.sum(image ^ circles)
        return cost

    
    def optimize(self) -> List[Tuple[
            int, int, Size]]:
        """
        Solves the packing problem and returns the center coords and 
        radius of each circle
        """
        # check for trivial case
        if self.is_rec and len(self.R)==1:
            step = np.max(self.R)
            kernel = np.ones(self.R[0])

            # recreate the images using the squares
            centers = signal.convolve2d(self.image, 
                kernel[::-1, ::-1], 
                mode='valid')[::step, ::step]
            centers[centers >= 1] = 1
            centers = [(j*step, i*step, self.R[0]) 
                for i, row in enumerate(centers)
                    for j, value in enumerate(row) if value]
            self.centers = centers.copy()
            return centers


        # real optimization starts here
        picked_centers = [] # initial solution
        while (center:= self.pick_center(picked_centers, 
                self.image)) is not None:
            picked_centers += [center]

        # cost of each circle
        c_costs = [self.out_cost(c, self.image) 
            for c in picked_centers]

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

            c_costs = [self.out_cost(c, self.image) 
                for c in n_picked_centers]
            if i%10 == 0:
                print(i)

        self.centers = n_picked_centers
        return n_picked_centers.copy()


    def make_image(self, scale: float,
            labeled_filters: np.ndarray) -> np.ndarray:
        """
        Creates a sample of the image after the optimization
        """
        m_images = {}

        if hasattr(self.R[0], '__iter__'):
            return "Unsopported"
        for r in self.R:
            h = int(2*scale*r)
            w = int(2*scale*r)
            m_images[r] = [
                cv2.resize(labeled_filter, (w+1, h+1))
                for labeled_filter in labeled_filters
            ]

        # create new image
        should_rotate = True
        synthetic_img = np.zeros((*self.image.shape, 4), dtype=np.uint8)
        # add the biggest images last, so that they stay on top
        # of the generated image
        self.centers.sort(key=lambda c: c[-1])
        for center in self.centers:
            x, y, r = center
            angle = np.random.randint(360) if should_rotate else 0
            m_img = random.choice(m_images[r])
            m_img = rotate(m_img, angle=angle, reshape=False)
            mask = self.calc_mask(x, y, scale*r+1)

            overlayed = utils.overlay(synthetic_img[mask], m_img)
            synthetic_img[mask] = overlayed.reshape(-1,4)

        return synthetic_img


    def save_solution(self, filename: str,
            image_names: Iterable[str], scale=1.0) -> None:
        """
        Saves the solution to be rendered by Open GL
        """
        WIDTH, HEIGHT = self.image.shape
        data = {str(i): {
                # normalize between -1 and 1
                "x": 2 * center[0] / WIDTH - 1,
                "y": 2 * center[1] / HEIGHT - 1,
                "r": center[2][0] / min(WIDTH, HEIGHT) \
                    if self.is_rec else center[2] / min(WIDTH, HEIGHT),
                "r2": center[2][1] / min(WIDTH, HEIGHT) \
                    if self.is_rec else 0,
                "filename": random.choice(image_names),
                "scale": scale
            }
            for i, center in enumerate(self.centers)}

        if not filename.startswith("config/"): 
            filename = "config/" + filename
        if not filename.endswith(".json"): 
            filename += ".json"
        
        with open(filename, "w") as f:
            json.dump(data, f)

