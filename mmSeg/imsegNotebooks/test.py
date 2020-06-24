#!/usr/bin/env python

import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from imageio import imread, imsave
from tqdm import tqdm

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization

# To output results in PAGE XML format (http://www.primaresearch.org/schema/PAGE/gts/pagecontent/2013-07-15/)
PAGE_XML_DIR = './page_xml'

# Transform probability maps into binary maps
def post_process_probs_ornament(probability_maps):

    binary_maps = np.zeros_like(probability_maps, np.uint8)
    binary_maps = np.delete(binary_maps, 0, 2)

    # Ornament
    binary_image = binarization.thresholding(probability_maps[:, :, 1], threshold=0.75)
    binary_image = binarization.cleaning_binary(binary_image, kernel_size=3)
    boxes = boxes_detection.find_boxes(binary_image, mode='rectangle', min_area=0.)
    bin_map = np.zeros_like(binary_maps)
    binary_maps[:, :, 0] = cv2.fillPoly(bin_map, boxes, (255, 0, 0))[:, :, 0]

    return binary_maps, boxes



def resize_image_coordinates(input_coordinates, input_shape, resized_shape):
    
    rx = input_shape[0] / resized_shape[0]
    ry = input_shape[1] / resized_shape[1]

    return np.stack((np.round(input_coordinates[:, 0] / ry),
                      np.round(input_coordinates[:, 1] / rx)), axis=1).astype(np.int32)

def format_quad_to_string(quad):
    """
    Formats the corner points into a string.
    :param quad: coordinates of the quadrilateral
    :return:
    """
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]

if __name__ == '__main__':

    # If the model has been trained load the model, otherwise use the given model
    model_dir = 'mark_model/export'
    if not os.path.exists(model_dir):
        model_dir = '../demo/model/'

    input_files = glob('prize_papers/mark/test/images/*')
    
    print(input_files)

    output_dir = 'processed_marks'
    os.makedirs(output_dir, exist_ok=True)
    # PAGE XML format output
    output_pagexml_dir = os.path.join(output_dir, PAGE_XML_DIR)
    os.makedirs(output_pagexml_dir, exist_ok=True)

    # Store coordinates of page in a .txt file
    txt_coordinates = ''

    with tf.Session():  # Start a tensorflow session
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')

        for filename in tqdm(input_files, desc='Processed files'):
            # For each image, predict each pixel's label
            print('Prediction')
            print(filename)
            name = filename[33:]
            print(name)
            
            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]
            original_shape = prediction_outputs['original_shape']
            
            # post-process predictions
            binary_map, boxes = post_process_probs_ornament(prediction_outputs['probs'][0])
            boxes_resized = [resize_image_coordinates(box, 
                                                      probs.shape[:2], 
                                                      original_shape) for box in boxes]
            
            imsave("prob_masks/" + name, probs[:,:,1])
            imsave("bin_masks/" + name, binary_map[:,:,0])
            
            # Draw page box on original image and export it. Add also box coordinates to the txt file
            original_img = imread(filename, pilmode='RGB')
            if boxes_resized is not None:
                for box in boxes_resized:
                    cv2.polylines(original_img, [box[:, None, :]], True, (0, 0, 255), thickness=5)
                # Write corners points into a .txt file
                txt_coordinates += '{},{}\n'.format(filename, format_quad_to_string(boxes_resized))

                # Create page region
                mark_border = PAGE.Border(coords=PAGE.Point.cv2_to_point_list(box[:, None, :]))
            else:
                print('No box found in {}'.format(filename))
                mark_border = PAGE.Border()

            basename = os.path.basename(filename).split('.')[0]
            imsave(os.path.join(output_dir, '{}_boxes.jpg'.format(basename)), original_img)
            
            # export
            text_regions = [PAGE.TextRegion(id='txt-reg-{}'.format(i), 
                                        coords=PAGE.Point.array_to_point(coords), 
                                        custom_attribute="structure{type:drop-cap;}") for i, coords in enumerate(boxes_resized)]

            page = PAGE.Page(image_filename=os.path.basename(filename),
                             image_height=original_shape[0],
                             image_width=original_shape[1],
                             text_regions=text_regions)
            
            

            page.write_to_file(filename=os.path.join(output_pagexml_dir, 
                                                     os.path.basename(name).split('.')[0] + '.xml'),
                               creator_name='MerchantMarkExtractor')
            
    # Save txt file
    with open(os.path.join(output_dir, 'pages.txt'), 'w') as f:
        f.write(txt_coordinates)
