import numpy as np
import matplotlib.pyplot as plt
from utils import *

# load image
img = np.loadtxt("teared_image")

# image size
K,L = img.shape
print(f"The image size is {K}x{L}")

# display image
f, ax = plt.subplots(1, 1)
ax.imshow(img, cmap="gray")
plt.show()

# indexes corresponding to tearing
indJ = np.where(img == 0)

print(f'Tearing represents {round(100*indJ[0].size/K/L,3)} % of the image\n')

# indexes of complemetary area
indI = np.nonzero(img)

# Parameters
nb_iterations = 5000
rho = 1  # Regularization parameter
tau = 0.01  # Step size for gradient descent

# Initialize restored image
x = np.copy(img)

for i in range(nb_iterations):
    v, h = grad(x)  # Compute gradients
    d = div(v, h)   # Compute divergence
    
    x -= tau * d  # Gradient descent step
    
    # Project onto known pixel values
    x = projballl2(x, img, rho, indI)


# Display final restored image
plt.imshow(x, cmap="gray")
plt.title("Restored Image")
plt.show()