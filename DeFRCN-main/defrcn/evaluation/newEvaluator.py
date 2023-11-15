from re import X
import time
import torch
import logging
import datetime
import numpy as np
from math import sqrt
import pandas as pd
import os
from contextlib import contextmanager
from .calibration_layer import PrototypicalCalibrationBlock
from defrcn.evaluation.comparator import OneShotLearning 
import math

one_shot_learner = OneShotLearning()
possible_lost_indices = []
file_data = []
frame_data_dict = {}
good_boxes = []
video_resolution = [1280,720]

def euclidean_distance(box1, box2):
    # Calculate the center of each box
    center1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
    center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
    
    if math.isnan(center1[0]) or math.isnan(center2[0]):
        return -1  # If there is a NaN box, return a very small value

    # Calculate the Euclidean distance between the centers
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(center1, center2)))

def check_target_update(score_threshold=0.6, drop_percentage=0.18, wait_frames=12):
    global possible_lost_indices
    global frame_data_dict
    global file_data

    # Add new data to frame_data_dict
    for data in file_data:
        frame = data[0]
        if frame not in frame_data_dict:
            frame_data_dict[frame] = []
        frame_data_dict[frame].append(data)
    
    # Get the highest final scored bounding box from the last frame
    last_frame_data = max(frame_data_dict[max(frame_data_dict.keys())], key=lambda x: x[5])
    _, _, _, _, _, _, _, similarity = last_frame_data

    # If there's a previous frame, compare the similarity score with the current frame
    if len(possible_lost_indices) > 0:
        # Retrieve the highest final scored bounding box's data for the previous frame
        prev_frame_data = max(frame_data_dict[possible_lost_indices[-1]], key=lambda x: x[5])
        _, _, _, _, _, _, _, prev_similarity = prev_frame_data

        # If the similarity score drops more than the drop_percentage, return True
        if prev_similarity != 0 and (prev_similarity - similarity) / prev_similarity > drop_percentage:
            return True
    
    # If the similarity score is below the threshold, add the frame to possible_lost_indices
    if similarity < score_threshold:
        possible_lost_indices.append(max(frame_data_dict.keys()))
    
    # Check the frames in possible_lost_indices
    for index in possible_lost_indices[:]:
        # If we have waited for enough frames
        if max(frame_data_dict.keys()) - index > wait_frames:
            # Check the last frame
            _, _, _, _, _, _, _, similarity = max(frame_data_dict[max(frame_data_dict.keys())], key=lambda x: x[5])
            if similarity > score_threshold:
                # Tracking regained, so no need to update
                possible_lost_indices.remove(index)
            else:
                # The scores didn't rise above the threshold
                possible_lost_indices.remove(index)
                return True  # Initiate update
    
    # If we've checked all possibilities and none required an update
    return False

def select_target_by_objectness(lookback_frames=4875, ignore_frames=0, punishment_factor=0.9999, start=0):
    global file_data
    global frame_data_dict 

    # Create a dict equivalent to frame_data_dict
    frame_data_dict = {}
    for data in file_data:
        frame_number = data[0]
        if frame_number not in frame_data_dict:
            frame_data_dict[frame_number] = []
        frame_data_dict[frame_number].append(data)

    if len(frame_data_dict) < ignore_frames + start:
        return None , None, None

    frame_keys = list(frame_data_dict.keys())[start:lookback_frames+start]

    # Sort the first frame boxes by objectness scores
    first_frame_boxes = sorted(frame_data_dict[frame_keys[0]], key=lambda x: 2*x[5] - x[7], reverse=True)

    num_groups = min(3, len(first_frame_boxes))  # Adjust the number of groups based on available boxes

    if num_groups == 0:
        return None, None, None

    best_boxes_group = {i: [] for i in range(num_groups)}

    for key in frame_keys:
        frame_boxes = frame_data_dict[key]
        for i in range(num_groups):
            closest_box = min(frame_boxes, key=lambda box: euclidean_distance(box[1:5], first_frame_boxes[i][1:5]))
            best_boxes_group[i].append((closest_box, key))

    # Calculate mean objectness scores for each group
    best_group = max(best_boxes_group.items(), key=lambda group: sum(2*box[5] - box[7] for box, frame in group[1]) / len(group[1]))[1]

    best_group = [x for x in best_group if x[1] >= frame_keys[0] + ignore_frames]  # Ignore the first frames

    # Determine the best box with punishment factor using objectness score
    best_box, best_frame = max(best_group, key=lambda x: (2*x[0][5] - x[0][7]) * punishment_factor ** (frame_keys[-1] - x[1]))

    # Check both similarity and objectness scores of the best box. If both are below 0.25, return None, None
    similarity_score = best_box[7]
    objectness_score = 2*best_box[5] - best_box[7]

    if similarity_score < 0.25 and objectness_score < 0.25:
        return None, None, None

    # Return the best box's coordinates, the frame it's from, and its objectness score adjusted with the punishment factor
    return best_box[1:5], best_frame, objectness_score * punishment_factor ** (frame_keys[-1] - best_frame)

def select_target_update_with_overlap_from_start(lookback_frames=4875, ignore_frames=0, punishment_factor=0.9999, start=0):
    global file_data
    global frame_data_dict 
    
    # Create a dict equivalent to frame_data_dict
    frame_data_dict = {}
    for data in file_data:
        frame_number = data[0]
        if frame_number not in frame_data_dict:
            frame_data_dict[frame_number] = []
        frame_data_dict[frame_number].append(data)

    if len(frame_data_dict) < ignore_frames + start:
        return None , None, None

    frame_keys = list(frame_data_dict.keys())[start:lookback_frames+start]

    # Sort the first frame boxes by final scores
    first_frame_boxes = sorted(frame_data_dict[frame_keys[0]], key=lambda x: x[5], reverse=True)

    num_groups = min(3, len(first_frame_boxes))  # Adjust the number of groups based on available boxes

    if num_groups == 0:
        return None, None, None

    best_boxes_group = {i: [] for i in range(num_groups)}

    for key in frame_keys:
        frame_boxes = frame_data_dict[key]
        for i in range(num_groups):
            closest_box = min(frame_boxes, key=lambda box: euclidean_distance(box[1:5], first_frame_boxes[i][1:5]))
            best_boxes_group[i].append((closest_box, key))

    # Calculate mean final scores without punishment
    best_group = max(best_boxes_group.items(), key=lambda group: sum(box[5] for box, frame in group[1]) / len(group[1]))[1]

    best_group = [x for x in best_group if x[1] >= frame_keys[0] + ignore_frames]  # Ignore the first frames

    # Determine the best box with punishment factor
    best_box, best_frame = max(best_group, key=lambda x: x[0][5] * punishment_factor ** (frame_keys[-1] - x[1]))

    # Check both similarity and objectness scores of the best box. If both are below 0.25, return None, None
    similarity_score = best_box[7]
    objectness_score = 2*best_box[5] - best_box[7]  # Correct calculation

    if similarity_score < 0.25 and objectness_score < 0.25:
        return None, None, None

    return best_box[1:5], best_frame, best_box[5]* punishment_factor ** (frame_keys[-1] - best_frame)

def get_max_index(tensor):
    max_val, max_idx = torch.max(tensor, 0)
    return max_idx.item()

def pad_array(array, length):
    if length <= len(array):
        return array  # No padding required

    num_zeros = length - len(array)
    padded_array = array + [0] * num_zeros
    return padded_array

def check_for_jumps(main_object):
    data = pd.DataFrame(file_data, columns=['frame', 'xmin', 'ymin', 'width', 'height', 'conf', 'class', 'prob_before_pcb'])
    data = data.tail(200)
    data = data.sort_values(['frame', 'conf'], ascending=[True, False]).drop_duplicates('frame')
    data['x_center'] = data['xmin'] + data['width'] / 2
    data['y_center'] = data['ymin'] + data['height'] / 2
    video_resolution = [1280, 720]
    data['euclidean_dist'] = (data[['x_center', 'y_center']].diff().pow(2).sum(axis=1))**0.5
    last_two_frames = data.tail(2)
    video_hypotenuse = np.hypot(*video_resolution)

    for index, row in last_two_frames.iterrows():
        bbox_hypotenuse = np.hypot(row['width'], row['height'])
        jump_threshold = max(video_resolution) * (bbox_hypotenuse / video_hypotenuse) / 2
        if row['euclidean_dist'] > jump_threshold:
          return True, not(main_object)
    return False, main_object

def get_groundtruth(frame_number, video_name):
    # Set up the path to the ground truth file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    groundtruth_file = os.path.join(os.path.dirname(os.getcwd()), "groundtruths", f"groundtruth{video_name}.txt")

    # Initialize the last valid bounding box and its frame number
    last_valid_box = None
    last_valid_frame = None

    # Open the ground truth file
    with open(groundtruth_file, 'r') as file:
        lines = file.readlines()

        # Loop over the lines in the file in reverse
        for idx, line in enumerate(lines[:frame_number][::-1]):
            # Convert string of coordinates to a list of integers
            box_coords = list(map(float, line.strip().split(',')))

            # Check if the bounding box is valid (i.e., not Nan and all dimensions > 0)
            if not np.isnan(box_coords).any() and all(x > 0 for x in box_coords):
                last_valid_box = list(map(int, box_coords))
                last_valid_frame = frame_number - idx
                break

    # Return the last valid bounding box and the frame number
    return last_valid_box, last_valid_frame, 1

def extract_bbox_coordinates(pcb):
    bbox_coordinates = []  # List to store all bounding box coordinates

    # Get the total number of bounding boxes
    num_boxes = pcb.get("pred_boxes").tensor.shape[0]

    # Loop over each bounding box
    for count in range(num_boxes):
        xmin = int(pcb.get("pred_boxes").tensor[count][0].item())
        ymin = int(pcb.get("pred_boxes").tensor[count][1].item())
        xmax = int(pcb.get("pred_boxes").tensor[count][2].item())
        ymax = int(pcb.get("pred_boxes").tensor[count][3].item())
        
        bbox_coordinates.append((xmin, ymin, xmax-xmin, ymax-ymin))  # Append the bounding box coordinates to the list

        # If there are more than 5 boxes, remove the extra ones from the end
        if len(bbox_coordinates) > 5:
            bbox_coordinates.pop()

    # If there are no boxes at all, fill bbox_coordinates with dummy data
    if len(bbox_coordinates) == 0:
        bbox_coordinates = [(0, 0, 0, 0)] * 5

    # If there are less than 5 boxes, duplicate the last box
    while len(bbox_coordinates) < 5:
        bbox_coordinates.append(bbox_coordinates[-1])

    return bbox_coordinates

def process_pcb(pcb, idx, before_pcb_scores):
    global file_data
    number_of = pcb.get("scores").size(0)
    count = wrong_count = 0
    #relations = pad_array(relations, number_of)  
    while(count < number_of):
        if(pcb.get("pred_classes")[count].item() == 0):
            wrong_count += 1
            x1 = int(pcb.get("pred_boxes").tensor[count][0].item())
            y1 = int(pcb.get("pred_boxes").tensor[count][1].item())
            x2 = int(pcb.get("pred_boxes").tensor[count][2].item())
            y2 = int(pcb.get("pred_boxes").tensor[count][3].item())
            prob = float(pcb.get("scores")[count].item()) if pcb.get("scores").size(0) == 1 else float(pcb.get("scores").data[count].item())
            prob_before_pcb = before_pcb_scores[count]
            file_data.append([idx, x1, y1, x2-x1, y2-y1, prob, pcb.get("pred_classes")[count].item(), prob_before_pcb])
        else:
            file_data.append([0,0,0,0,0,0,0,0])
        count += 1
    if number_of == 0:
            file_data.append([0,0,0,0,0,0,0,0])

def inference_on_dataset_n(model, data_loader, last_change, cfg, videoname,  groundtruth, interval, threshold, bestf,bestb, objectness):
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    jump_frame = -1
    pcb = None
    one_shot_learner.initialize_sample()

    main_object, in_frame = True, True
    logger.info("Start initializing PCB module, please wait a seconds...")
    pcb = PrototypicalCalibrationBlock(cfg)
    total = len(data_loader)
    logging_interval = 50
    num_warmup = 0
    start_time = time.time()
    total_compute_time = 0

    with (inference_context(model)):
      with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx <= last_change:
                continue
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            outputs , scores = pcb.execute_calibration_2(inputs, outputs)
            after_pcb = outputs[0].get("instances")
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            #bboxes = extract_bbox_coordinates(after_pcb)
            #path = "datasets/VOCCustom/JPEGImages/" + str(idx).zfill(8) + ".jpg"
            #one_shot_learner.initialize_test(bboxes, path)
            #relations =  one_shot_learner.find_relations()
            #relations = relations.tolist()
            process_pcb(after_pcb, idx, scores)
            #print(file_data)
            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(seconds=int(seconds_per_img * (total - num_warmup) - duration))
                logger.info("Inference done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta)))

            if idx % interval == 0 and idx != 0:  #check_target_update():
                
                if groundtruth:
                    if objectness:
                        best_box, best_frame,score = select_target_by_objectness(punishment_factor=threshold, start = bestf + 1)
                    else:        
                        best_box, best_frame,score = select_target_update_with_overlap_from_start(punishment_factor=threshold, start = bestf + 1)
                    if best_box != None:
                      best_box,best_frame,score = get_groundtruth(best_frame, videoname)
                    else:
                        best_box = bestb
                        best_frame = bestf
                        same = True
                  
                else:  
                    if objectness:
                        best_box, best_frame,score = select_target_by_objectness(punishment_factor=threshold, start = bestf + 1)
                    else:        
                        best_box, best_frame,score = select_target_update_with_overlap_from_start(punishment_factor=threshold, start = bestf + 1)
                    if best_box == None:
                        best_box = bestb
                        best_frame = bestf
                        same = True
      
                print("Target update at: {}\n".format(idx))

                return best_frame, best_box, idx, score
                
    with open('results_pcb_tu.txt', 'a') as f:
      for video_info in file_data:
          f.write("{},{},{},{},{},{},{},{}\n".format(*video_info))
    print("Video end")
    return -1, [-1,-1,-1,-1], -1, -1

def is_in():
  return True
    
@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)