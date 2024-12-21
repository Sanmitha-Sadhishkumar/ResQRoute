"""#Binary grid representation of image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def hurd_convert(image_source):
    # Load the image and convert it into a numpy array
    img = Image.open(image_source) #"../static/show/ground floor.png"
    img_gray = img.convert('L')  #Graysacle
    numpydata = np.asarray(img_gray)
    threshold_value = 127  #Pixel threshold for differentiating

# Binary representation
    binary_grid = np.where(numpydata > threshold_value, 1, 0)
    print(binary_grid)
    grid_height, grid_width = binary_grid.shape

# Displaying
    plt.imshow(binary_grid, cmap='gray')
    plt.title("Binary Grid Representation")
    plt.axis('off')

# Save the binary image
    plt.savefig('../static/show/binaryimage.png', bbox_inches='tight', pad_inches=0)
    plt.close()   # Free up memory
    print("Binary grid of image saved")

# Save the binary grid as a NumPy file 
    np.save('../static/show/binary_grid.npy', binary_grid)
    plt.figure(figsize=(8, 8))
    obstacles = np.argwhere(binary_grid == 0)
    # Plot obstacles as blue squares
    for obs in obstacles:
        plt.plot(obs[1], obs[0], 'bs', markersize=5)  # Blue squares for obstacless
    plt.xlim(0, grid_width)
    plt.ylim(0, grid_height)
    plt.gca().invert_yaxis() #Invert to match image coordinates
    plt.title("Obstacles Representation")
    plt.axis('on')

    # Save the obstacles plot
    plt.savefig('../static/show/plotimage.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Obstacle plotted image saved")
    
"""
#Binary grid representation of image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def hurd_convert(image_source):
# Load the image and convert it into a numpy array
    img = Image.open(image_source)
    img_gray = img.convert('L')  #Graysacle
    numpydata = np.asarray(img_gray)
    threshold_value = 127  #Pixel threshold for differentiating

# Binary representation
    binary_grid = np.where(numpydata > threshold_value, 1, 0) 
    grid_height, grid_width = binary_grid.shape # 1 for walkable, 0 for non-walkable

# Displaying
    plt.imshow(binary_grid, cmap='gray')
    plt.title("Binary Grid Representation")
    plt.axis('off')

# Save the binary image
    plt.savefig('../static/show/binaryimage.png', bbox_inches='tight', pad_inches=0)
    plt.close()   # Free up memory
    print("Binary grid of image saved")

# Save the binary grid as a NumPy file
    np.save('../static/show/binary_grid.npy', binary_grid)

#Plotting obstacles

    def plot_obstacles(binary_grid):
        plt.figure(figsize=(8, 8))
        obstacles = np.argwhere(binary_grid == 0)
    # Plot obstacles as blue squares
        for obs in obstacles:
            plt.plot(obs[1], obs[0], 'bs', markersize=5)  # Blue squares for obstacless
        plt.xlim(0, grid_width)
        plt.ylim(0, grid_height)
        plt.gca().invert_yaxis() #Invert to match image coordinates
        plt.title("Obstacles Representation")
        plt.axis('on')

        # Save the obstacles plot
        plt.savefig('../static/show/plotimage.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        print("Obstacle plotted image saved")

    plot_obstacles(binary_grid)
    result =''''''
    for i in binary_grid:
        for j in i:
            result+=str(j)
            result+=' '
        result+='\n'
    return result