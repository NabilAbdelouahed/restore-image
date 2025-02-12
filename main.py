import numpy as np
import matplotlib.pyplot as plt
from utils import *

# load image
img = read_image_and_get_pixels("teared_image.png")

# image size
K,L = img.shape
print(f"The image size is {K}x{L}")


# display image
f, ax = plt.subplots(1, 1)
ax.imshow(img, cmap="gray")
plt.show()

# indices corresponding to tearing
indJ = np.where(img == 0)

print(f'Tearing represents {100*len(indJ[0])/K/L} % of the image\n')

# indices of complemetary area
indI = np.nonzero(img)