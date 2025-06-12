from skimage.feature import canny
from skimage.color import lab2rgb, rgb2gray, rgb2lab
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
import concurrent.futures

class EdgeMethod():
  def __init__(self, canny_sigma = 0.7):
    self.canny_sigma = canny_sigma
  
  def __call__(self, img):
    return canny(rgb2gray(img), sigma=self.canny_sigma).astype('float32')

def lab_separate(img, l_factor = 1, l_midpoint=50, a_factor = 1, a_midpoint=50, b_factor = 1, b_midpoint = 50):
    lab = rgb2lab(img)
    lab[:,:,0] = lab[:,:,0] + l_factor*(lab[:,:,0] - l_midpoint*np.ones_like(lab[:,:,0]))
    lab[:,:,1] = lab[:,:,1] + a_factor*(lab[:,:,1] - a_midpoint*np.ones_like(lab[:,:,1]))
    lab[:,:,2] = lab[:,:,2] + b_factor*(lab[:,:,2] - b_midpoint*np.ones_like(lab[:,:,2]))
    over = lab[:,:,0] > 100
    under = lab[:,:,0] < 0
    lab[:,:,0][over] = 100
    lab[:,:,0][under] = 0
    return lab2rgb(lab)

class GrapheneEdgeMethod(EdgeMethod):
  def __call__(self, img):
    e1 = canny(rgb2gray(lab_separate(img, l_factor=1.5, a_factor=1, b_factor=2, l_midpoint=50, a_midpoint=-47, b_midpoint=-50)), sigma=1.75)
    e1_5 = canny(rgb2gray(lab_separate(img, 1.5, l_midpoint=50)), sigma=1.4)
    e2 = canny(rgb2gray(lab_separate(img, 2, l_midpoint=50)), sigma=0.5)
    e2b = canny(rgb2gray(lab_separate(img, 2, l_midpoint=60)), sigma=1.4)
    e2total = np.logical_and(e2, e2b)
    combined = np.logical_or(e1_5, e2total)
    combined = np.logical_or(combined, e1)
    return combined
  

def subdivide(image, n):
    """
    Splits the input image into n^2 (as equal as possible) subsections,
    even if h or w is not divisible by n.

    Args:
        image (np.ndarray): Input image (2D or 3D array).
        n (int): Number of subdivisions per axis.

    Returns:
        list: A list of n^2 image subsections (as numpy arrays).
    """
    h, w = image.shape[:2]
    h_indices = [round(i * h / n) for i in range(n + 1)]
    w_indices = [round(i * w / n) for i in range(n + 1)]
    sections = []
    for i in range(n):
        for j in range(n):
            section = image[h_indices[i]:h_indices[i+1], w_indices[j]:w_indices[j+1]]
            sections.append(section)
    return sections

def combine_sections(sections, n, original_shape=None):
    """
    Combines a list of n^2 image subsections into the original image,
    even if h or w is not divisible by n.

    Args:
        sections (list): List of n^2 numpy arrays (subsections).
        n (int): Number of subdivisions per axis.
        original_shape (tuple, optional): The original image shape (h, w) or (h, w, c).
                                          If not provided, it will be inferred.

    Returns:
        np.ndarray: The reconstructed image.
    """
    # Infer original shape if not provided
    if original_shape is None:
        # Estimate shape from sections and n
        h_total = sum([sections[i * n].shape[0] for i in range(n)])
        w_total = sum([sections[i].shape[1] for i in range(n)])
        if sections[0].ndim == 3:
            c = sections[0].shape[2]
            image = np.zeros((h_total, w_total, c), dtype=sections[0].dtype)
        else:
            image = np.zeros((h_total, w_total), dtype=sections[0].dtype)
    else:
        image = np.zeros(original_shape, dtype=sections[0].dtype)

    h_offsets = [0]
    w_offsets = [0]
    # Calculate offsets for rows
    for i in range(n):
        h_offsets.append(h_offsets[-1] + sections[i * n].shape[0])
    # Calculate offsets for columns
    for j in range(n):
        w_offsets.append(w_offsets[-1] + sections[j].shape[1])

    idx = 0
    for i in range(n):
        for j in range(n):
            h0, h1 = h_offsets[i], h_offsets[i+1]
            w0, w1 = w_offsets[j], w_offsets[j+1]
            image[h0:h1, w0:w1] = sections[idx]
            idx += 1
    return image

class EntropyEdgeMethod(EdgeMethod):
  def __init__(self, k=3, magnification=20):
    self.k = k
    self.mag = magnification
    

  def __call__(self, img):
    '''
    Input 2K or 4K resolution images for best results.
    '''

    input = rgb2gray(img)
    sections = subdivide(input, self.k)
    if self.mag <= 10:
       disk_radius = 4
       percentile_threshold = 85
    else:
       disk_radius = 15
       percentile_threshold = 75

    footprint = disk(disk_radius)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(entropy, section, footprint) for section in sections]
      entropied_sections = [f.result() for f in futures]

    entropied = combine_sections(entropied_sections, self.k)
    entropied = (entropied/np.max(entropied))
    entropied = np.pow(entropied, 2.5)

    threshold = np.percentile(entropied, percentile_threshold) * 1.25
    return entropied > threshold

class Material():
  def __init__(self, name, target_bg_lab, layer_labels, sigma=0.7, fat_threshold=0.1, Edge_Method = EdgeMethod):
    self.name = name
    self.target_bg_lab = target_bg_lab
    self.layer_labels = layer_labels
    self.recommended_sigma = sigma
    self.recommended_fat_threshold = fat_threshold
    self.Edge_Method = Edge_Method




# Make each material

wte2_labels = { # cielab colorspace
                  (56, 40, -10): 'monolayer',
                  (46, 50, -20): 'bilayer',
                  (39, 60, -30): 'trilayer',
                  (31, 54, -32): 'fewlayer',
                  (70, 20, -30): 'manylayer',
                  (80, 5, 10): 'bulk',
                  (0, 0, 0): 'dirt',
                  (30, 20, -10): 'dirt',
              }

wte2 = Material('WTe2', [58.50683594, 28.57762527, -2.79295444], layer_labels=wte2_labels)

graphene_labels = { # cielab colorspace
                  (49.5, 14, 4): 'monolayer',
                  (47, 17, 3): 'bilayer',
                  (37, 31, -2): 'trilayer',
                  (30, 30, -27): 'fewlayer',
                  (50, 0, -15): 'manylayer',
                  (80, 5, 10): 'bulk',
                  (53, -8, -12): 'bluish_layers',
                  (50, 1, -10): 'bluish_layers',
                  (48.1, 4.7, 10.4): 'dirt',
                #   (30, 20, -10): 'dirt',
                  (0,0,0): 'bg'
              }

graphene = Material(
    'Graphene',
    [49.62793933, 12.17574073,  5.43972594],
    layer_labels=graphene_labels,
    sigma=3,
    fat_threshold=0.005,
    Edge_Method = EntropyEdgeMethod,
  )


hBN_labels = { # cielab colorspace
                  (37.4, 16.6, -8.3): '0-10',
                  (43.9, 25.6, -64.8): '10-20',
                  (62.2, -11.6, -41.3): '20-30',
                  (72.8, -35, -15.1): '30+',
                  (71.8, -25.5, 0): '35+',
                  # (69.4, -3.5, 50.8): 'bulk',
                  # (53, -8, -12): 'bluish_layers',
                  # (50, 1, -10): 'more_bluish_layers',
                  (30, 20, -10): 'dirt',
              }

hBN = Material(
    'hBN', 
    [39.43335796, 14.2159274,  -4.54227961], 
    layer_labels=hBN_labels, 
    sigma=0.7, 
    fat_threshold=0.1,
    Edge_Method=EntropyEdgeMethod,
  )