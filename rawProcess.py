import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt

raw_filename = 'import.ARW'

with rawpy.imread(raw_filename) as raw:
    
    print(f'raw type:                     {raw.raw_type}')                      # raw type (flat or stack, e.g., Foveon sensor)
    print(f'number of colors:             {raw.num_colors}')                    # number of different color components, e.g., 3 for common RGB Bayer sensors with two green identical green sensors 
    print(f'color description:            {raw.color_desc}')                    # describes the various color components
    print(f'raw pattern:                  {raw.raw_pattern.tolist()}')          # decribes the pattern of the Bayer sensor
    print(f'black levels:                 {raw.black_level_per_channel}')       # black level correction
    print(f'white level:                  {raw.white_level}')                   # camera white level
    print(f'color matrix:                 {raw.color_matrix.tolist()}')         # camera specific color matrix, usually obtained from a list in rawpy (not from the raw file)
    print(f'XYZ to RGB conversion matrix: {raw.rgb_xyz_matrix.tolist()}')       # camera specific XYZ to camera RGB conversion matrix
    print(f'camera white balance:         {raw.camera_whitebalance}')           # the picture's white balance as determined by the camera
    print(f'daylight white balance:       {raw.daylight_whitebalance}')         # the camera's daylight white balance

    # get raw image data
    image = np.array(raw.raw_image, dtype=np.double)
    print(image.shape)
    # subtract black levels and normalize to interval [0..1]
    black = np.reshape(np.array(raw.black_level_per_channel, dtype=np.double), (2, 2))
    black = np.tile(black, (image.shape[0]//2, image.shape[1]//2))
    image = (image - black) / (raw.white_level - black)
    # find the positions of the three (red, green and blue) or four base colors within the Bayer pattern
    n_colors = raw.num_colors
    colors = np.frombuffer(raw.color_desc, dtype=np.byte)
    pattern = np.array(raw.raw_pattern) 
    index_0 = np.where(colors[pattern] == colors[0])
    index_1 = np.where(colors[pattern] == colors[1])
    index_2 = np.where(colors[pattern] == colors[2])
    index_3 = np.where(colors[pattern] == colors[3])
    # apply white balance, normalize white balance coefficients to the 2nd coefficient, which is usually the coefficient for green
    wb_c = raw.camera_whitebalance 
    wb = np.zeros((2, 2), dtype=np.double) 
    wb[index_0] = wb_c[0] / wb_c[1]
    wb[index_1] = wb_c[1] / wb_c[1]
    wb[index_2] = wb_c[2] / wb_c[1]
    if n_colors == 4:
        wb[index_3] = wb_c[3] / wb_c[1]
    wb = np.tile(wb, (image.shape[0]//2, image.shape[1]//2))
    image_wb = np.clip(image * wb, 0, 1)
    # demosaic via downsampling
    image_demosaiced = np.empty((image_wb.shape[0]//2, image_wb.shape[1]//2, n_colors))
    if n_colors == 3:
        image_demosaiced[:, :, 0] = image_wb[index_0[0][0]::2, index_0[1][0]::2]
        image_demosaiced[:, :, 1]  = (image_wb[index_1[0][0]::2, index_1[1][0]::2] + image_wb[index_1[0][1]::2, index_1[1][1]::2]) / 2
        image_demosaiced[:, :, 2]  = image_wb[index_2[0][0]::2, index_2[1][0]::2]
    else: # n_colors == 4
        image_demosaiced[:, :, 0] = image_wb[index_0[0][0]::2, index_0[1][0]::2]
        image_demosaiced[:, :, 1] = image_wb[index_1[0][0]::2, index_1[1][0]::2]
        image_demosaiced[:, :, 2] = image_wb[index_2[0][0]::2, index_2[1][0]::2]
        image_demosaiced[:, :, 3] = image_wb[index_3[0][0]::2, index_3[1][0]::2]
    # convert to linear sRGB, calculate the matrix that transforms sRGB into the camera's primary color components and invert this matrix to perform the inverse transformation
    XYZ_to_cam = np.array(raw.rgb_xyz_matrix[0:n_colors, :], dtype=np.double)
    sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], dtype=np.double)
    sRGB_to_cam = np.dot(XYZ_to_cam, sRGB_to_XYZ)
    norm = np.tile(np.sum(sRGB_to_cam, 1), (3, 1)).transpose()
    sRGB_to_cam = sRGB_to_cam / norm
    if n_colors == 3:
        cam_to_sRGB = np.linalg.inv(sRGB_to_cam)
    else: # n_colors == 4
        cam_to_sRGB = np.linalg.pinv(sRGB_to_cam)
    image_sRGB = np.einsum('ij,...j', cam_to_sRGB, image_demosaiced)  # performs the matrix-vector product for each pixel
    # apply sRGB gamma curve
    i = image_sRGB < 0.0031308
    j = np.logical_not(i)
    image_sRGB[i] = 323 / 25 * image_sRGB[i]
    image_sRGB[j] = 211 / 200 * image_sRGB[j] ** (5 / 12) - 11 / 200
    image_sRGB = np.clip(image_sRGB, 0, 1)
    # show image
    plt.axis('off')
    plt.imshow(image_sRGB)
    plt.savefig("example.png", dpi=1200)
    plt.close('all')
    #plt.waitforbuttonpress()

    plt.hist(image_sRGB.ravel(), bins=256, range=(0.0, 1.0))
    #plt.hist(image_sRGB.ravel(), 256, [0,256])
    plt.show()
    plt.waitforbuttonpress()
    #plt.imsave('name.png', image_sRGB, dpi=300)
    