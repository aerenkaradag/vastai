from ctypes import Array
import os
# import numpy as np
import xml.etree.ElementTree as ET
import cv2
import shutil
from random import randint
import linecache
import argparse
import glob
import random

def dogrudizim(number):
    return str(number).zfill(6)

def get_object_dimensions(tree, object_name):
    # Iterate over all 'object' elements in the XML tree
    for obj in tree.iter('object'):
        # If the object's name matches the given name
        if obj.find('name').text == object_name:
            # Extract and return the bounding box dimensions
            bndbox = obj.find('bndbox')
            width = int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)
            height = int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
            return width, height
    # If no matching object was found, return None
    return None

def add_background_object(tree, image_width, image_height, object_area):
    # Calculate a random width and height for the new object, ensuring the area matches
    new_width = random.randint(1, image_width)
    new_height = int(object_area / new_width)

    # Calculate random coordinates for the new object
    xmin = 1#random.randint(0, image_width - new_width)
    ymin = 1#random.randint(0, image_height - new_height)
    xmax = xmin + new_width
    ymax = ymin + new_height

    # Create the new 'object' element and its child elements
    new_obj = ET.SubElement(tree.getroot(), 'object')
    ET.SubElement(new_obj, 'name').text = "background"
    ET.SubElement(new_obj, 'pose').text = "Unspecified"
    ET.SubElement(new_obj, 'truncated').text = "0"
    ET.SubElement(new_obj, 'difficult').text = "0"
    bndbox = ET.SubElement(new_obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(xmin)
    ET.SubElement(bndbox, 'ymin').text = str(ymin)
    ET.SubElement(bndbox, 'xmax').text = str(xmax)
    ET.SubElement(bndbox, 'ymax').text = str(ymax)

def add_background_to_xml(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)

    # Get the image width and height from the XML
    size = tree.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    # Get the dimensions of the 'secondnew' object
    dimensions = get_object_dimensions(tree, 'secondnew')

    if dimensions is not None:
        # Calculate the area of the 'secondnew' object
        width, height = dimensions
        object_area = width * height

        # Add a new object with the name "background" to the XML
        add_background_object(tree, image_width, image_height, object_area)

        # Write the changes back to the XML file
        tree.write(xml_path)
    else:
        print(f"No object named 'secondnew' found in {xml_path}")
        
def fotoyuGetir(number,x1,x2,y1,y2,video):
        
    print("results are: {}, {},{},{},{}".format(number, x1,x2,y1,y2))
    one_above = os.path.dirname(os.getcwd())
    src = os.path.join(one_above, "sample.xml")
    
    dest = os.path.join(os.getcwd(), "datasets", "VOC2007")
    dest = os.path.join(dest, "Annotations", str(dogrudizim(9975)) + ".xml")
    
    img_path = os.path.join(os.path.join(one_above,"video_resim"), video, str(str(number).zfill(8)) + ".jpg")
    img_dest = os.path.join(os.path.join(os.getcwd(), "datasets", "VOC2007"), "JPEGImages", str(dogrudizim(9975)) + ".jpg")
    print("img_path: {}, img_dest: {}".format(img_path,img_dest))
    img = cv2.imread(os.path.join(img_path))
    height, width = img.shape[0], img.shape[1]
    shutil.copyfile(img_path, img_dest)
    
    tree = ET.parse(src)
    root = tree.getroot()
    root = root.find("size")
    root.find("width").text = str(width)
    root.find("height").text = str(height)
    tree.write(src)
    shutil.copy(src,dest)
    
    
    tree = ET.parse(dest)
    root = tree.getroot()
    print(dogrudizim(number) + ".jpg")
    root.find("filename").text = dogrudizim(9975) + ".jpg"
    root.find("source").find("flickrid").text = str(randint(0, 100000))
    root = root.find("object")
    root.find("name").text = "secondnew"
    root = root.find("bndbox")
    root.find("xmin").text = str(x1)
    root.find("ymin").text = str(x2)
    root.find("xmax").text = str(x1 + y1)
    root.find("ymax").text = str(x2 + y2)
    tree.write(dest)
    add_background_to_xml(dest)
    
    