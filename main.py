import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
 
print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread("clouds.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
cv2.imshow("Over the Clouds", img)
cv2.imshow("Over the Clouds - gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()


raw_filename = 'A1_07396.arw'
 
with rawpy.imread(raw_filename) as raw:
    print(f'raw type:                     {raw.raw_type}')                      # raw type (flat or stack, e.g., Foveon sensor)
    print(f'number of colors:             {raw.num_colors}')                    # number of different color components, e.g., 3 for common RGB Bayer sensors with two green identical green sensors 
    print(f'color description:            {raw.color_desc}')                    # describes the various color components
    print(f'raw pattern:                  {raw.raw_pattern.tolist()}')          # decribes the pattern of the Bayer sensor
    print(f'black levels:                 {raw.black_level_per_channel}')       # black level correction
    print(f'white level:                  {raw.white_level}')                   # camera white level
    print(f'color matrix:                 {raw.color_matrix.tolist()}')         # camera specific color matrix, usually obtained from a list in rawpy (not from the raw file)
    print(f'XYZ to RGB conversion matrix: {raw.rgb_xyz_matrix.tolist()}')       # camera specific XYZ to camara RGB conversion matrix
    print(f'camera white balance:         {raw.camera_whitebalance}')           # the picture's white balance as determined by the camera
    print(f'daylight white balance:       {raw.daylight_whitebalance}')         # the camera's daylight white balance

with rawpy.imread(raw_filename) as raw:
    image = raw.raw_image
    rgb = raw.postprocess(rawpy.Params(use_camera_wb=True))
    plt.imshow(rgb)
    plt.axis('off')

def adjust_blacklevel(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def fix_orientation(image, orientation):
    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image    