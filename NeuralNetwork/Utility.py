import numpy as np

def CropData(data, shape):
    offsets = [(data.shape[1] - shape[1]) // 2, (data.shape[2] - shape[2]) // 2]
    
    return data[:, offsets[0]:-offsets[0], offsets[1]:-offsets[1]]