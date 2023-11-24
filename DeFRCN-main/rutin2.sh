#!/bin/bash

# The directory where the script will be executed
WORKING_DIR="/home/vastai/DeFRCN-main"

# List of video names to test. Replace this array with your actual list of video names.
declare -a video_names=("agility" "animal" "ants1" "bag" "ball2" "ball3" "basketball" "birds1" "birds2" "bolt1" "book" "bubble" "butterfly" "car1" "conduction1" "crabs1" "dinosaur" "diver" "drone1" "drone_across" "fernando" "fish1" "fish2" "flamingo1" "frisbee" "girl" "graduate" "gymnastics1" "gymnastics2" "gymnastics3" "hand" "hand2" "handball1" "handball2" "helicopter" "iceskater1" "iceskater2" "kangaroo" "lamb" "leaves" "marathon" "matrix" "monkey" "motocross1" "nature" "polo" "rabbit" "rabbit2" "rowing" "shaking" "singer2" "singer3" "snake" "soccer1" "soccer2" "soldier" "surfing" "tennis" "tiger" "wheel" "wiper" "zebrafish1")

# Helper function to run the tests
run_test() {
    local script_name=$1
    local video_name=$2
    local interval=$3 # target update interval
    local groundtruth=$4
    local objectness=$5
    local iterations=$6
    local saved_model=$7  # new variable for saved model path

    cd /home/vastai
    python3 fotocekici.py --custom --count 1 --random True --video $video_name --cevat

    cd $WORKING_DIR

    cmd="python3 $script_name --interval $interval --num-gpus 1 --video $video_name --threshold 0.999"

    if [ "$groundtruth" = "true" ]; then
        cmd="$cmd --groundtruth"
    fi

    if [ "$objectness" = "true" ]; then
        cmd="$cmd --objectness"
    fi

    cmd="$cmd --config-file configs/voc/targetupdate2.yaml \
        --opts OUTPUT_DIR gecici/$video_name \
        SOLVER.MAX_ITER $iterations"

    # Use the saved model for mainistan2.py tests, otherwise use model_reset_remove.pth
    if [[ "$script_name" == "mainistan2.py" && -n "$saved_model" ]]; then
        cmd="$cmd MODEL.WEIGHTS $saved_model"
    else
        cmd="$cmd MODEL.WEIGHTS /home/vastai/model_reset_remove.pth"
    fi

    cmd="$cmd TEST.PCB_ENABLE True \
        TEST.PCB_MODELPATH data/pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth"

    eval $cmd
}

# Running the tests for each video name
for video in "${video_names[@]}"; do
    printf "Video: %s\n" "$video"
    cd /home/vastai
    python3 fotocekici.py --custom --count 1 --random True --video $video
    # Test 1 with mainistan2.py
    run_test "mainistan2.py" $video 30000 false false 800 ""
    saved_model_path="/home/vastai/DeFRCN-main/gecici/$video/model_final.pth"
    
    # # Test 2 with mainistan2.py
    # run_test "mainistan2.py" $video 240 true false 800 $saved_model_path
    # # Test 3 with mainistan2.py
    # run_test "mainistan2.py" $video 240 true true 800 $saved_model_path
    # # Test 4 with mainistan2.py
    # run_test "mainistan2.py" $video 240 false true 800 $saved_model_path
    # # Test 5 with mainistan2.py
    # run_test "mainistan2.py" $video 240 false false 800 $saved_model_path
    # # Test 6 with mainistan.py
    # run_test "mainistan.py" $video 240 true false 80 ""
    # # Test 7 with mainistan.py
    # run_test "mainistan.py" $video 240 true true 80 ""
    # # Test 8 with mainistan.py
    # run_test "mainistan.py" $video 240 false true 80 ""
    # Test 9 with mainistan.py
    # run_test "mainistan.py" $video 240 false false 800 ""

    #videoda random access, random bi frameden target update ile başlatma ve sona gelmeden durma (belli bir aralıkta çalışma)
done