"""
Reconstructs an image using a given shape.
The reconstruction is done as if solving a packing
problem
"""

import cv2
import json
import random
import joblib
import numpy as np
from PIL import Image

import scipy.signal as signal
from scipy.ndimage.interpolation import rotate

from typing import (Any, Dict, Iterable, 
    List, Tuple, Union)

from . import utils


# writing union all the time 
Size = Union[int, np.ndarray]

class PackingOptimizer:
    """
    Solves a packing problem to fill the target image
    with circles.
    """

    def __init__(self, R: Iterable[Size], 
            image: np.ndarray, rotate_90=False) -> None:
        self.centers: List[Tuple[int, int, Size]]
        self.initial_solution: List[Tuple[int, int, Size]]

        self.image = image.copy() if len(image.shape) == 2 \
            else utils.load_bin_image(image, True)

        self.is_rec = hasattr(R[0], '__iter__')
        if self.is_rec:
            self.R = [np.array(r, dtype=np.int16) for r in R]
            if rotate_90: self.R += [np.array(r[::-1], dtype=np.int16) for r in R]
            self.calc_mask = lambda x, y, r: utils.mrec_mask(
                self.image.shape)(x, y, r[0], r[1])
        
        else:
            self.R = R.astype(np.int16) # circles radius
            # lambda so joblib works
            self.calc_mask = lambda x, y, r: utils.mcircle_mask(
                self.image.shape)(x, y, r)

        self._AREA = np.sum(self.image == 1)
        

    def __call__(self, n=10, p=0.1) -> \
            List[Tuple[int, int, int]]:
        return self.optimize(n, p)


    @classmethod
    def from_images(cls, imgs_path: Iterable[str],
            scales: Iterable[float], image: np.ndarray,
            rotate_90=False) -> "PackingOptimizer":
        """
        Creates the optimizer using the sizes from images
        """
        shapes = [Image.open(img_path).size
            for img_path in imgs_path]
        R = [(int(s[0] * sc), int(s[1] * sc))
                for s in shapes 
                    for sc in scales]

        return cls(R, image, rotate_90)
        

    @property
    def cost(self) -> int:
        """
        Sum of circle's area outside the image and gaps 
        between circles
        """
        return self.total_cost(self.centers, self.image)

    @property
    def inner_cost(self) -> int:
        """
        Area inside the image not filled by circles
        """
        mask = np.array(
            [self.calc_mask(*p) for p in self.centers]
        ).sum(0)
        return np.sum(self.image & ~mask)

    @property
    def outer_cost(self) -> int:
        """
        Area outside the image filled by circles
        """
        return self.cost - self.inner_cost


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
            try:
                R_copy = R_copy[:next((i 
                    for i, r in enumerate(R_copy) if r is center[2]))]    
            except StopIteration:
                R_copy = []
        else:
            R_copy = R_copy[:R_copy.index(center[2])]

        if len(R_copy) == 0:
            return (picked_centers, False)
        
        while (center:= self.pick_center(picked_centers, 
                image, R_copy)) is not None:
            picked_centers.append(center)
        
        return (picked_centers, True)


    def out_cost(self, center: Tuple[int, int, Size]) \
            -> int:
        """
        Area of the given object outside the image
        """
        return self.area_outside(self.image, *center)


    def area_outside(self, img: np.ndarray, cx: int, 
            cy: int, r: Size) -> int:
        """
        Returns the area outside the image
        """
        obj_area = self.calc_mask(cx, cy, r)
        return np.sum(obj_area | img) - self._AREA


    def total_cost(self, 
            picked_centers: List[Tuple[int, int, Size]],
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


    def greedy_solution(self, _picked_centers=[]):
        """
        Uses a greedy approach to solve the packing
        problem
        """
        # initial solution
        picked_centers = _picked_centers.copy()
        while (center:= self.pick_center(picked_centers, 
                self.image)) is not None:
            picked_centers += [center]

        # cost of each circle
        c_costs = [self.out_cost(c) 
            for c in picked_centers]

        # optimize circles which are outside the image
        n_picked_centers = picked_centers

        while True:
            m = np.argmax(c_costs)
            _n_picked_centers, changed = self.minimize_circle_loss(
                n_picked_centers[m], self.image, n_picked_centers)
            
            if changed:
                n_picked_centers = _n_picked_centers
            else:
                break

            c_costs = [self.out_cost(c) 
                for c in n_picked_centers]

        return n_picked_centers


    def random_removal(self, 
            centers: List[Tuple[int, int, Size]],
            p: float) -> List[Tuple[int, int, Size]]:
        """
        Removes p% of points from the centes
        """
        k = int((1-p) * len(centers))
        return random.sample(centers, k)


    def repair(self, centers: List[Tuple[int, int, Size]],
            n :int, best_cost: int) -> List[Tuple[int, int, Size]]:
        """
        Recreates points of the solution after removal
        """
        def parallel_greedy(centers):
            n_picked_centers = self.greedy_solution(centers)
            cost = self.total_cost(n_picked_centers, self.image)
            return (n_picked_centers, cost)

        results = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(parallel_greedy)(centers) for _ in range(n))
        centers, cost = min(results, key=lambda x: x[1])
        
        if cost < best_cost:
            self.centers = centers
            print(f"\t(Repair) Best cost: {cost}")

        return self.centers

    
    def optimize(self, n=100, p=0.1) -> List[Tuple[
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

        print("Greedy algorithm start")
        # real optimization starts here
        # run greedy algorithm for n tries
        self.repair([], n, np.inf)

        print("Greedy algorithm done")

        self.initial_solution = self.centers.copy()

        for _n in range(5 * n):
            n_centers = self.random_removal(self.centers, p)
            print(f"Starting iter {_n} with p = {p}")
            self.repair(n_centers, n, self.cost)

        return self.centers.copy()


    def get_dict(self, image_names: Iterable[str], 
            scale=1.0) -> Dict[str, Dict[str, Any]]:
        """
        Returns the data in a dict format
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
        
        return data

    def save_solution(self, filename: str,
            image_names: Iterable[str], scale=1.0) -> None:
        """
        Saves the solution to be rendered by Open GL
        """
        data = self.get_dict(image_names, scale)
        
        if not filename.startswith("config/"): 
            filename = "config/" + filename
        if not filename.endswith(".json"): 
            filename += ".json"
        
        with open(filename, "w") as f:
            json.dump(data, f)



class SecPackingOptimizer:
    """
    Sections the image in different parts and creates
    a packing optimizer for each one.
    """
    def __init__(self, n_sections: int, Rs: Iterable[Iterable[Size]], 
            image: np.ndarray, rotate_90=False) -> None:
        self.centers: List[Tuple[int, int, Size]]

        if type(rotate_90) == bool:
            rotate_90 = [rotate_90] * n_sections

        sections = utils.vsection_image(image, n_sections)
        self.opts = [PackingOptimizer(Rs[n], 
            sections[n], rotate_90[n]) 
            for n in range(n_sections)]

    
    def __call__(self, ns=10, ps=0.1) -> \
            List[Tuple[int, int, int]]:
        """
        Calls every optimizer
        """
        if type(ns) == int: ns = [ns] * len(self.opts)
        if type(ps) == float or type(ps) == int:
            ps = [ps] * len(self.opts)
        centers = [opt(ns[i], ps[i]) 
            for i, opt in enumerate(self.opts)]
        self.centers = [c for cnt in centers for c in cnt]
        return self.centers.copy()

    
    def save_solution(self, filename: str,
            image_names: Iterable[Iterable[str]], scale=1.0) -> None:
        """
        Saves the solution to be rendered by Open GL
        """
        data_list = [opt.get_dict(image_names[i]) 
            for i, opt in enumerate(self.opts)]

        data = {str(j + i * len(dt.values())): d
            for i, dt in enumerate(data_list)
                for j, d in enumerate(dt.values())}

        if not filename.startswith("config/"): 
            filename = "config/" + filename
        if not filename.endswith(".json"): 
            filename += ".json"
        
        with open(filename, "w") as f:
            json.dump(data, f)

        return data