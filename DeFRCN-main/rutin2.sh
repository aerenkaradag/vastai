#!/bin/bash

# The directory where the script will be executed
WORKING_DIR="/home/vastai/DeFRCN-main"

# List of video names to test. Replace this array with your actual list of video names.
declare -a video_names=("bolt1")

# Helper function to run the tests
run_test() {
    local script_name=$1
    local video_name=$2
    local interval=$3
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

    cd /home/vastai
    python3 fotocekici.py --custom --count 1 --random True --video $video
    # # Test 1 with mainistan2.py
    # short term videolarında 0 + 2 (base+novel)
    # run_test "mainistan2.py" $video 30000 false false 800 ""
    # saved_model_path="/home/vastai/DeFRCN-main/gecici/$video/model_final.pth"
    
    #### my test
    saved_model_path="/home/vastai/DeFRCN-main/gecici/$video/model_final.pth"
    run_test "mainistan2.py" $video 3000 true true 800 $saved_model_path

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
    #T1, T6, T9
    #videoda random access, random bi frameden target update ile başlatma ve sona gelmeden durma (belli bir aralıkta çalışma)
done