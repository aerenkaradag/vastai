import torch
from torchvision import transforms
from PIL import Image
from defrcn.evaluation.comparator import OneShotLearning  # assuming the above class is saved in one_shot_learning.py file

# Path to your images
test_image_path = "datasets/VOCCustom/JPEGImages/" + str(1).zfill(8) + ".jpg"

# Bounding boxes as tuples in the form (xmin, ymin, width, height)
bounding_boxes_sample = [(789, 489, 41, 107),(789, 489, 41, 107),(789, 489, 41, 107),(789, 489, 41, 107),(789, 489, 41, 107)]  # xmin, ymin, width, height
bounding_boxes_test = [(790, 488, 38, 105), (885, 436, 37, 109), (718, 437, 32, 102), (758, 428, 36, 83), (758, 428, 36, 83)]

one_shot_learner = OneShotLearning()
one_shot_learner.initialize_sample()
one_shot_learner.initialize_test(bounding_boxes_test, test_image_path)
print(bounding_boxes_test)
print(test_image_path)
relations=one_shot_learner.find_relations()



print(relations)
