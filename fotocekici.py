from ctypes import Array
import os
# import numpy as np
import xml.etree.ElementTree as ET
import cv2
import shutil
import random
from random import randint
import linecache
import argparse
import glob
parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, required=True)
parser.add_argument('--cevat', action='store_true')
parser.add_argument('--random', type=bool, required=False)
parser.add_argument('--sirali', type=bool, required=False)
parser.add_argument('--custom', action='store_true')
parser.add_argument('--no-custom', dest='feature', action='store_false')
parser.add_argument('--video', type=str, required = True)
#parser.add_argument('--resimsayisi', type=int, required = True)

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
    xmin = 3#random.randint(0, image_width - new_width)
    ymin = 3#random.randint(0, image_height - new_height)
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

def sayi_sec(gercekresim):
    return randint(1, gercekresim)

def dogrusayi(sayi):
  src = os.path.join("groundtruths","groundtruth{}.txt".format(args.video)) 
  count = 0
  a = 0
  file1 = open(src, 'r')
  Line = file1.readlines()
  #print(len(Line))
  print("sayi: {}".format(sayi))
  for lines in Line:
    txt = lines.split(',')
    a = a + 1
    if(txt[0] != "nan" and int(float(txt[0])) != 0):
      count = count + 1
    if count == sayi:
      print("dogru sayi: {}".format(a))
      return a




def sayi_sec1(gercekresim, sayi):
  a = 0
  count = 0
  while a < gercekresim:
    a = a + 1
    txt = linecache.getline(os.path.join("groundtruths","groundtruth{}.txt".format(args.video)), a)
    txt = txt.split(",")
    if(txt[0] != "nan" and int(float(txt[0])) != 0):
      count = count + 1
  
  resimnumarasi = 1
  c = []
  print("count: {}".format(count))
  b = 0

  while b < args.count:
    resimnumarasi = int(resimnumarasi + ((count - 1) / sayi))
    print(resimnumarasi)
    c.append(dogrusayi(resimnumarasi))
    b = b + 1
  assert None not in c
  return c


def dogrudizim(number):
    if (number < 10):
        return "0000000" + str(number)
    elif (number < 100):
        return "000000" + str(number)
    elif (number < 1000):
        return "00000" + str(number)
    elif (number < 5000):
        return "0000" + str(number)
    else:
        return "00" + str(number)


def okumaca(number):
    src = os.path.join(os.getcwd(), "sample.xml")
    src_emp = os.path.join(os.getcwd(), "sample_empty.xml")
    c = []
    num = 1
    a = 0
    while (number > 0):
        a = a + 1
        txt = linecache.getline(os.path.join("groundtruths","groundtruth{}.txt".format(args.video)), a)
        
        txt = txt.split(",")
        #,print(int(float(txt[0])) == 0)
        if(str(txt[0]) != "nan" and int(float(txt[0])) != 0):
          #print(a)
          c.append(a)
          txt[0] = (int(float(txt[0])))
          txt[1] = (int(float(txt[1])))
          txt[2] = (int(float(txt[2])))
          txt[3] = (int(float(txt[3][:-1])))
          dest = os.path.join("VOCCustom", "Annotations", dogrudizim(num) + ".xml")
          shutil.copyfile(src, dest)
          tree = ET.parse(dest)
          root = tree.getroot()
          root.find("filename").text = dogrudizim(num) + ".jpg"
          root.find("source").find("flickrid").text = str(randint(0, 100000))
          root = root.find("object")
          root.find("name").text = "secondnew"
          root = root.find("bndbox")
          root.find("xmin").text = str(txt[0])
          root.find("ymin").text = str(txt[1])
          root.find("xmax").text = str(txt[0] + txt[2])
          root.find("ymax").text = str(txt[1] + txt[3])
          tree.write(dest)
          num = num + 1
        else: 
          c.append(a)
          dest = os.path.join("VOCCustom", "Annotations", dogrudizim(num) + ".xml")
          
          shutil.copyfile(src_emp, dest)
          tree = ET.parse(dest)
          root = tree.getroot()
          root.find("filename").text = dogrudizim(num) + ".jpg"
          root.find("source").find("flickrid").text = str(randint(0, 100000))
          tree.write(dest)
          num = num + 1
        number = number - 1
    print(num)    
    return c, num 


def ad_ayarla(number, c):
    num = 1
    a = 0
    f = open('6.txt', 'w')
    print("gen: {}".format(len(c)))
    while (number > a):
        #print(c[a])
        src = os.path.join("video_resim", args.video ,str(c[a]).zfill(8) + ".jpg")
        dest = os.path.join("VOCCustom", "JPEGImages", str(dogrudizim(num)) + ".jpg")

        f.write(str(dogrudizim(num)) + "\n")
        shutil.copyfile(src, dest)
        a = a + 1
        num = num + 1
def main(args):
    num = 9975
    count = 0
    if args.cevat:
        dest = os.path.join(os.getcwd(), "DeFRCN-main")
        dest = os.path.join(dest, "datasets", "VOC2007")
        for number in [1]:
          #print("girdigirdi")
          src = os.path.join("DeFRCN-main", "datasets", os.path.join("VOCCustom", "JPEGImages", str(dogrudizim(number)) + ".jpg"))
          dst = os.path.join(dest, "JPEGImages", str(dogrudizim(num + count) + ".jpg"))
          shutil.copyfile(src, dst)

          src = os.path.join("DeFRCN-main", "datasets", os.path.join("VOCCustom", "Annotations", str(dogrudizim(number)) + ".xml"))
          dst = os.path.join(dest, "Annotations", str(dogrudizim(num+ count)) + ".xml")
          shutil.copyfile(src,dst)

          num = num + 1
          count = count + 1
    else:
      src = os.path.join(os.getcwd(), "sample.xml")
      src_emp = os.path.join(os.getcwd(), "sample_empty.xml")
      
      img = cv2.imread(os.path.join("video_resim", args.video,"00000001.jpg"))
      
      height, width = img.shape[0], img.shape[1]
      tree = ET.parse(src)
      tree2 = ET.parse(src_emp)
      root = tree.getroot()
      root2 = tree2.getroot()
      root = root.find("size")
      root2 = root2.find("size")
      root.find("width").text = str(width)
      root.find("height").text = str(height)
      root2.find("width").text = str(width)
      root2.find("height").text = str(height)
      tree.write(src)
      tree2.write(src_emp)

      resimsayisi = len(glob.glob1(os.path.join("video_resim", args.video),"*.jpg")) #args.resimsayisi
      print("resimsayisi = {}".format(resimsayisi))
      rand = 0	
      num = 9975
      foto = 0
      foto = args.count
      if args.custom:
        if os.path.isdir("VOCCustom"):
            shutil.rmtree("VOCCustom")
        os.mkdir("VOCCustom")
        os.mkdir("VOCCustom/Annotations")
        os.mkdir("VOCCustom/JPEGImages")
        os.mkdir("VOCCustom/ImageSets")
        os.mkdir("VOCCustom/ImageSets/Main")
        print("yaptım")
        c , numberss = okumaca(resimsayisi)
        ad_ayarla(numberss - 1 , c)
        shutil.copyfile("6.txt", os.path.join(os.path.join(os.getcwd(), "VOCCustom", "ImageSets" ), "Main", "6.txt"))
        os.rename(os.path.join(os.path.join(os.getcwd(), "VOCCustom", "ImageSets" ), "Main", "6.txt"),os.path.join(os.path.join(os.getcwd(), "VOCCustom", "ImageSets" ), "Main", "9.txt") )
        shutil.move("6.txt", os.path.join(os.path.join(os.getcwd(), "VOCCustom", "ImageSets" ), "Main"))
      if foto != 0:
          a = 0


          dest = os.path.join(os.getcwd(), "DeFRCN-main")
          dest = os.path.join(dest, "datasets", "VOC2007")
          path = os.path.join("DeFRCN-main","datasets", "VOCCustom")
          if os.path.isdir(path):
              shutil.rmtree(path)
          dest2 = os.path.join("DeFRCN-main", "datasets","VOCCustom")
          shutil.move("VOCCustom",dest2)
    
          if args.random:
              rand = [1] #sayi_sec(args.count)
          if args.sirali:
              print("buradayım")
              rand = sayi_sec1(resimsayisi,args.count) 
          print(rand)
          
          count = 0
          for number in rand:
            #print("girdigirdi")
            src = os.path.join("DeFRCN-main", "datasets", os.path.join("VOCCustom", "JPEGImages", str(dogrudizim(number)) + ".jpg"))
            dst = os.path.join(dest, "JPEGImages", str(dogrudizim(num + count) + ".jpg"))
            shutil.copyfile(src, dst)

            src = os.path.join("DeFRCN-main", "datasets", os.path.join("VOCCustom", "Annotations", str(dogrudizim(number)) + ".xml"))
            dst = os.path.join(dest, "Annotations", str(dogrudizim(num+ count)) + ".xml")
            shutil.copyfile(src,dst)

            num = num + 1
            count = count + 1
      #add_background_to_xml(dst)
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
