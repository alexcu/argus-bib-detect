#!/usr/bin/env python3

"""
 Script to preprocess OCR output for Tesseract

 Usage:
 python3 preprocess.py /path/to/input/dir \
                       /path/to/output/dir
"""

from glob import glob
import os
import shutil
import sys
import cv2
import numpy as np

def preprocess(img):
    """Takes a given image and returns the preprocessed version for
    tesseract.

    Args:
        img (cv2 image): The image to preprocess
    Returns
        cv2 image: The preprocessed image.
    """

    # gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray scale inverted
    img_gray_inv = cv2.bitwise_not(img_gray)
    # blurring to increase edges and color contrast:'gaussian convolution' with a kernel
    img_blur = cv2.GaussianBlur(img_gray_inv,ksize=(5,5),sigmaX=0,sigmaY=0)

    # statistical flag for white versus black ID: median
    flag_background = np.median(img_gray)

    # initial idea: 128-ish is half the size of the RGB scale so:
    if flag_background > 128:
        print('White Number Detected!')
        _, threshold = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV)
    else:
        print('Black Number Detected!')
        _, threshold = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # Return the processed image image
    return threshold

def multi_preprocess(img):
    """
    Produces candidate images for input to tesseract
    input:
        img (array)
    output: dictionary of processed images with the following keys
        'color' : original image
        'gray'  : gray scale image
        'gray_inv': inversion of the gray scale image (white <--> black)
        'blur_gauss': gaussian blurring (convolution),
        'blur_gauss_inv': inversion of the gaussian blurring (white <--> black),
        'blend': sketch transformation
        'thresh': threshold version
        'eq_hist': equalization histogram
    Suggestion from exploratory phase: consider gray and gray inv as the top two options.
    If the quality of the bib is high, all the previous choices generate positive results
    """
    # gray scale conversion
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray scale inverted
    img_gray_inv = cv2.bitwise_not(img_gray)
    
    # equalization histogram
    eq_hist = cv2.equalizeHist(img_gray)
    
    #blurring to increase edges and color contrast:'gaussian convolution' with a kernel
    # in this case the filter (5,5) works fine and the sigma parameters extracted
    # from the image. That is why they are set to zero
    img_blur = cv2.GaussianBlur(img_gray,ksize=(5,5),sigmaX=0,sigmaY=0)
    
    # inversion
    img_blur_inv = cv2.bitwise_not(img_blur)
    
    # statistical flag for white versus black ID: mean works better than median
    # this will only make sense for the cropped bib digits region
    flag_background = np.mean(img_gray)
    # RGB scale values:
    # black=(0,0,0)   white=(255,255,255)
    # initial idea: 128-ish is half the size of the RGB scale so:
    if flag_background > 128:
        print('White Number')
        ret1 , th1 = cv2.threshold(img_gray,220,255,cv2.THRESH_BINARY_INV)
    else:
        print('Black Number')
        ret1 , th1 = cv2.threshold(img_gray,128,255,cv2.THRESH_BINARY)
        
    # blending the image with its convolution: sketch transformation
    # this was useful when the bib region was much bigger
    # probably not so useful when the bib region is much smaller
    img_blend = cv2.divide(img_gray_inv,255-img_blur,scale=256)
    
    # output: ideally a very clean threshold version is the input for tesseract
    # although if the bib is clean the gray scale images can be fed into
    # tesseract also with positive results
    dic_images = {'color': img ,
                  'gray': img_gray,
                  'gray_inv': img_gray_inv,
                  'blur_gauss': img_blur,
                  'blur_gauss_inv':img_blur_inv,
                  'blend': img_blend,
                  'thresh': th1,
                  'eq_hist': eq_hist}
    
    return dict_images

# usage:
#-- generating dictionary
#dic_images = multi_process(img)
#-- calling the top two candidate images from the dictionary
#img_tesseract_input_gray = dic_images['gray']
#img_tesseract_input_gray_inv = dic_images['gray_inv']

def main():
    assert len(sys.argv) - 1 >= 2, "Must provide two arguments (in_dir, out_dir)"

    in_dir = sys.argv[1]
    assert in_dir != None, "Missing input directory (argv[1])"

    out_dir = sys.argv[2]
    assert out_dir != None, "Missing output directory (argv[2])"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in glob("%s/*.jpg" % in_dir):
        print("Processing '%s' for thresholding..." % file)
        img = cv2.imread(file)
        image_id = os.path.splitext(os.path.basename(file))[0]
        out_jpeg_file = ("%s/%s.jpg" % (out_dir, image_id))
        cv2.imwrite(out_jpeg_file, preprocess(img))

    for file in glob("%s/*.json" % in_dir):
        image_id = os.path.splitext(os.path.basename(file))[0]
        out_json_file = ("%s/%s.json" % (out_dir, image_id))
        shutil.copy(file, out_json_file)

if __name__ == '__main__':
    main()
