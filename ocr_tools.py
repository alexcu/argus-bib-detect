# python tools for recovering the bib number given a bib region
import os
import pandas as pd
import cv2

# apply tesseract inside a python script, with the tesseract arguments
# from the exploratory phase, we recommend pre-process the input image for tesseract into its gray scale versions
# and use the psm argument with value 11 and limit the characters_do_detect
def tesseract_direct_bib( path2image , psm_argument , oem_argument, characters_to_detect ):
    """
    function to apply tesseract directly to the processed bib image
    inputs
    path2image (string)          : path to the bib image file
    psm_argument (int)           : page segmentation mode parameter 
                                     7 for single text line 
                                     8 for single word (recommended if the text region is highly constrained)
                                    10 for single character
                                    11 for sparse text (recommended for bib regions)
    oem_argument (int)           : ocr engine mode, default is 3. For tesseract version 4 dev
                                   4 means Neural Network (possible for the future)
    characters_to_detect (string): use '0123456789' if you only want to detect digits
    
    output convention:
    if the input is image.jpg the detections will be inside image_ocr_detections.txt
    """
    # generating a label for the output with the format
    # input:  path/to/image.jpg
    # output: path/to/image_ocr_detections.txt
    
    # split the string from the dot
    label_image = path2image.split('.')
    
    # extract the name label
    label_image = label_image[-2]
    
    # filename format to save
    fname2save = label_image + '_ocr_detections.txt'
    
    # command to apply the detector to an input image
    bash_command = 'tesseract' + ' ' + path2image + ' ' + \
                   'stdout -c tessedit_char_whitelist='+ characters_to_detect + ' ' + \
                   '--psm' + ' ' + str(psm_argument) + ' ' + \
                   '--oem' + ' ' + str(oem_argument) + ' ' + \
                   'makebox' + ' ' + '>' + ' ' + fname2save
    print('Running:' , bash_command)
    os.system(bash_command)


# function to build the bib number from tesseract output, including graphic output for the bib
def build_bib( output_file_tesseract , bib_image , output_image ):
    # reading output file from OCR as a dataframe
    df = pd.read_csv(output_file_tesseract , sep=" " , 
                     header=None,names=['id','x1','y1','x2','y2','page'])

    # note:
    # indexing by removing the false detections: labeled as '~' in the tesseract output
    # we focus only in numerical characters for this purpose
    # because with cropped bib images this was not present
    # dfc = df.loc[ df['id'] != '~' ]
    # this was more common in the whole bib region image and not present in the cropped bib
    dfc = df

    # to reset the indexing do:
    dfc = dfc.reset_index(drop=True)

    # mapping to array, and taking only the columns with label,x,y,x,y
    dfc_array = dfc.as_matrix()
    dfc_array = dfc_array[:,0:5]
    
    # sort array from left to right x coordinate (column 1 in python indexing) 
    # for bib reconstruction purposes
    dfc_array = dfc_array[ dfc_array[:,1].argsort() ]
    
    # remember this tesseract convention:
    # the 4 columns represent
    # left x , bottom y , right x , top y
    # but tesseract considers the origin in the lower left corner

    # image to display detections
    img_dets_tess = bib_image.copy()
    font = cv2.FONT_HERSHEY_COMPLEX
    digits_ID = []
    for label,c1,c2,c3,c4 in dfc_array:
        # adapting tesseract output into (x,y,w,h) format
        x = c1
        y = img_dets_tess.shape[0] - c4 # convert the origin from lower left to upper right
        # w,h parameters
        w = c3 - c1
        h = c4 - c2
        # perimeter, area and w/h ratio of the character
        peri_wh = 2 * (w+h)
        area_wh = w*h
        wh_ratio = 1.0 * w/h
        
        # note: in the exploration phase, when the bib region is cropped at its maximum 
        # there was no need to remove false positives, which was the case when the
        # bib region was much bigger. If this is the case, the recommendation to remove
        # the presence of false positives is to include an if statement 
        # with the wh_ratio values and a minimum area resolution
        #if (wh_ratio < 0.9) & (wh_ratio > 0.5) & (area_wh > 400):
        
        # display of detections
        cv2.rectangle(img_dets_tess,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText( img_dets_tess , str(label) , (x,y-10) , font , 1.0,(0,255,0) , 2 )
        
        # filling list of digit detections on the image
        digits_ID.append(label)
    
    # method to stack horizontally the digit detections to generate bib
    bibN = ''.join(str(digit) for digit in digits_ID)
    
    # mapping the bib number to integer just for utility
    bibN = int(bibN)
    
    # graphical display of the bib
    cv2.putText( img_dets_tess , 'bib:' + str(bibN) , (5,img_dets_tess.shape[0]-10) , font , 1.0,(0,255,0) , 2 )
    
    # saving output image with bib display
    cv2.imwrite(output_image,img_dets_tess)
    
    return bibN

