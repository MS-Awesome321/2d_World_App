import cv2
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

class Segmenter():
    def __init__(self, img, material, colors=None, numbers=None, min_area = 10, max_area = 1000000, magnification=100, k=3):
        self.img = img
        self.size = img.shape[:2]
        self.target_bg_lab = material.target_bg_lab
        self.layer_labels = material.layer_labels
        self.edge_method = material.Edge_Method(k=k, magnification=magnification)
        self.colors = colors
        self.numbers = numbers
        self.min_area = min_area
        self.max_area = max_area
        
        self.lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    def make_masks(self):
        self.edges = self.edge_method(self.img)
        self.masks, self.num_masks = label(np.logical_not(self.edges))
        self.mask_ids, self.mask_areas = np.unique(self.masks, return_counts=True)

        # Get BG mask
        self.bg_mask_id = np.argmax(self.mask_areas)

    def get_all_avg_lab(self):
        # Flatten masks and lab arrays
        flat_masks = self.masks.ravel()
        flat_lab = self.lab.reshape(-1, 3)
        num_masks = self.num_masks + 1

        # Prepare output array
        avg_labs = np.zeros((num_masks, 3), dtype=np.float32)

        # Compute sum for each channel using bincount, then divide by counts
        for c in range(3):
            sums = np.bincount(flat_masks, weights=flat_lab[:, c], minlength=num_masks)
            # Avoid division by zero
            avg_labs[:, c] = sums / self.mask_areas

        self.avg_labs = avg_labs
        return avg_labs

    def adjust_layer_labels(self):
        self.avg_bg_lab = self.avg_labs[self.bg_mask_id]

        adjustment_factor = self.avg_bg_lab - np.array(self.target_bg_lab)
        new_layer_labels = {}
        for key, value in self.layer_labels.items():
            new_key = tuple(key + adjustment_factor)
            new_layer_labels[new_key] = value
        
        self.layer_labels = new_layer_labels
        # Add bg lab
        self.layer_labels[tuple(self.avg_bg_lab)] = 'bg'

    def label_masks(self):
        self.adjust_layer_labels()
        self.mask_labels = []

        # Prepare layer label LABs and types as arrays/lists
        base_labs = np.array(list(self.layer_labels.keys()))
        layer_types = list(self.layer_labels.values())

        # Compute all pairwise distances (masks x base_labs)
        dists = np.linalg.norm(self.avg_labs[:, None, :] - base_labs[None, :, :], axis=2)
        min_indices = np.argmin(dists, axis=1)

        for idx, i in enumerate(self.mask_ids):
            area = self.mask_areas[i]
            if i==0:
                # Don't label edge mask
                self.mask_labels.append('bg')
            elif area < self.min_area:
                self.mask_labels.append('dirt')
            else:
                label = layer_types[min_indices[idx]]
                self.mask_labels.append(label)

    def process_frame(self):
        self.make_masks()
        self.get_all_avg_lab()
        self.label_masks()

    def prettify(self):
        # Prepare a lookup table for colors for all mask ids
        color_table = np.zeros((np.max(self.mask_ids) + 1, 3))
        # Assign colors for valid masks, black for others
        for idx, i in enumerate(self.mask_ids):
            area = self.mask_areas[idx]
            label = self.mask_labels[idx]
            if (area > self.min_area and area < self.max_area):
                color_table[i] = self.colors[label]
            else:
                color_table[i] = np.array([0, 0, 0])

        # Map each pixel in self.masks to its color using the lookup table
        colored_masks = color_table[self.masks]
        self.colored_masks = colored_masks
        return colored_masks

    def numberify(self):
        # Prepare a lookup table for numbers for all mask ids
        number_table = np.zeros(np.max(self.mask_ids) + 1)
        for idx, i in enumerate(self.mask_ids):
            area = self.mask_areas[idx]
            label = self.mask_labels[idx]
            if (area > self.min_area and area < self.max_area):
                number_table[i] = self.numbers[label]
            else:
                number_table[i] = 0

        # Map each pixel in self.masks to its number using the lookup table
        result = number_table[self.masks]
        self.numbered_masks = result
        return result