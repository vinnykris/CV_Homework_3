import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
        
    c_minX = 0
    c_maxX = img.shape[0]-1
    c_maxY = img.shape[1]-1
    c_minY = 0
    
    bottomRightX = ((M[0,0]*c_minX) + (M[0,1]*c_maxY) + M[0,2]) / ((M[2,0]*c_minX) + (M[2,1]*c_maxY) + M[2,2])
    bottomRightY = ((M[1,0]*c_minX) + (M[1,1]*c_maxY) + M[1,2]) / ((M[2,0]*c_minX) + (M[2,1]*c_maxY) + M[2,2])
    
    bottomLeftX = ((M[0,0]*c_minX) + (M[0,1]*c_minY) + M[0,2]) / ((M[2,0]*c_minX) + (M[2,1]*c_minY) + M[2,2])
    bottomLeftY = ((M[1,0]*c_minX) + (M[1,1]*c_minY) + M[1,2]) / ((M[2,0]*c_minX) + (M[2,1]*c_minY) + M[2,2])
    
    topRightX = ((M[0,0]*c_maxX) + (M[0,1]*c_maxY) + M[0,2]) / ((M[2,0]*c_maxX) + (M[2,1]*c_maxY) + M[2,2])
    topRightY = ((M[1,0]*c_maxX) + (M[1,1]*c_maxY) + M[1,2]) / ((M[2,0]*c_maxX) + (M[2,1]*c_maxY) + M[2,2])
    
    topLeftX =  ((M[0,0]*c_maxX) + (M[0,1]*c_minY) + M[0,2]) / ((M[2,0]*c_maxX) + (M[2,1]*c_minY) + M[2,2])
    topLeftY = ((M[1,0]*c_maxX) + (M[1,1]*c_minY) + M[1,2]) / ((M[2,0]*c_maxX) + (M[2,1]*c_minY) + M[2,2])
    
    minX = min(bottomRightX,bottomLeftX,topRightX,topLeftX)
    minY = min(bottomRightY,bottomLeftY,topRightY,topLeftY)
    maxX = max(bottomRightX,bottomLeftX,topRightX,topLeftX)
    maxY = max(bottomRightY,bottomLeftY,topRightY,topLeftY)
        
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)

def applyTransformation(x, y, M):
    x_transformed = ((M[0,0]*x) + (M[0,1]*y) + M[0,2]) / ((M[2,0]*x) + (M[2,1]*y) + M[2,2])
    y_transformed = ((M[1,0]*x) + (M[1,1]*y) + M[1,2]) / ((M[2,0]*x) + (M[2,1]*y) + M[2,2])
    return x_transformed, y_transformed

def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    warped_img = cv2.warpPerspective(img, M, (maxX - minX, maxY - minY))
    inverse_homography = np.linalg.inv(M)

    for x in range(warped_img.shape[0]):
        for y in range(warped_img.shape[1]):
            x_transformed, y_transformed = applyTransformation(x, y, inverse_homography)
            if x_transformed < 0 or x_transformed > (img.shape[0] - 1):
                continue
            if y_transformed < 0 or y_transformed > (img.shape[1] - 1):
                continue
            # Left and Right side
            if (y - minY) > blendWidth or (maxY - y) > blendWidth:
                alpha = 1
            elif (y - minY) < blendWidth:
                alpha = float((y - minY)/blendWidth)
            elif (maxY - y) < blendWidth:
                alpha = float((maxY - y)/blendWidth)
            
            acc[x, y, :] += np.array((alpha*warped_img[x, y, 0], alpha*warped_img[x, y, 1], alpha*warped_img[x, y, 2], alpha))


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    # acc[:,:,0] = np.divide(acc[:,:,0], acc[:,:,3])
    # acc[:,:,1] = np.divide(acc[:,:,1], acc[:,:,3])
    # acc[:,:,2] = np.divide(acc[:,:,2], acc[:,:,3])
    # acc[:,:,3] = np.divide(acc[:,:,3], acc[:,:,3])

    # acc[:,:,0] = np.where(acc[:,:,0]==np.inf, 0, acc[:,:,0])
    # acc[:,:,1] = np.where(acc[:,:,1]==np.inf, 0, acc[:,:,1])
    # acc[:,:,2] = np.where(acc[:,:,2]==np.inf, 0, acc[:,:,2])
    # acc[:,:,3] = np.where(acc[:,:,3]==np.inf, 1, acc[:,:,3])

    # acc[:,:,0] = np.where(acc[:,:,0]==np.nan, 0, acc[:,:,0])
    # acc[:,:,1] = np.where(acc[:,:,1]==np.nan, 0, acc[:,:,1])
    # acc[:,:,2] = np.where(acc[:,:,2]==np.nan, 0, acc[:,:,2])
    # acc[:,:,3] = np.where(acc[:,:,3]==np.nan, 1, acc[:,:,3])

    for x in range(acc.shape[0]):
        for y in range(acc.shape[1]):
            if acc[x, y, 3] == 0:
                acc[x, y, :] = 0
                continue
            acc[x, y, 0] = np.divide(acc[x,y,0], acc[x,y,3])
            acc[x, y, 1] = np.divide(acc[x,y,1], acc[x,y,3])
            acc[x, y, 2] = np.divide(acc[x,y,2], acc[x,y,3])
    
    acc[:, :, 3] = 1


    return acc


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        minimum_x, minimum_y, maximum_x, maximum_y = imageBoundingBox(img, M)
        if minimum_x < minX:
            minX = minimum_x
        if minimum_y < minY:
            minY = minimum_y
        if maximum_x > maxX:
            maxX = maximum_x
        if maximum_y > maxY:
            maxY = maximum_y

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, outputWidth)
    #TODO-BLOCK-END
    # END TODO
    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

