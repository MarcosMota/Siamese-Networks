import random
from typing import Tuple
import numpy as np

def create_pairs_on_set(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ create a pair of labeled images

    Args:
        images (np.ndarray): Array of images
        labels (np.ndarray): [Array of Labels

    Returns:
        Tuple[np.ndarray, np.ndarray]: pair of images and label
    """

    def create_pairs(x, digit_indices) -> Tuple[np.ndarray, np.ndarray]:
        '''
            Positive and negative pair creation.
            Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
                
        return np.array(pairs), np.array(labels)


    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')
    
    return pairs, y