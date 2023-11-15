from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import inference_on_dataset_n
from defrcn.engine import default_argument_parser, default_setup
from defrcn.engine import Trainer as DefaultTrainer #Trainer çoklu eğitim için benim yazdığım sınıf
from defrcn.dataloader import build_detection_test_loader
import photoextractor
import os

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)    
    #######################################threshold####################
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    ################################################
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)

    return cfg


def main(args):
    f_pcb = open("results_pcb_tu.txt", "w")
    bbox = []
    change = []
    best_frame = 0
    old_frame =-1
    old_bbox = (0,0,0,0)
    f_pcb.close()
    if args.groundtruth:
        filename = f"groundtruth_{args.interval}_{args.video}_objectness{args.objectness}re.txt"
    else:
        filename = f"{args.threshold}_{args.interval}_{args.video}_objectness{args.objectness}re.txt"
    cfg = setup(args)
    args.resume = False
    last_change = 0
    while last_change != -1:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)

        if(list(bbox) != list(old_bbox) and best_frame != old_frame ):
          trainer.train()
          old_bbox = bbox 
          old_frame = best_frame

        
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.OUTPUT_DIR + "/model_final.pth", resume=args.resume
        )
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            
            data_loader = build_detection_test_loader(cfg, dataset_name)

            best_frame,bbox,last_change,score = inference_on_dataset_n(model, data_loader, 
            last_change, cfg, args.video, args.groundtruth, args.interval, args.threshold, best_frame,bbox,args.objectness)

            change.append([best_frame,bbox,score])

            if last_change != 0:
                x1,x2,y1,y2 = bbox
                if best_frame != -1:
                  photoextractor.fotoyuGetir(best_frame,x1,x2,y1,y2, args.video)
                else: 
                  print("error")

    with open(filename, 'w') as file:
      for item in change:
          file.write("%s\n" % item)
    os.rename("results_pcb_tu.txt", f"results_{args.video}_{args.groundtruth}_{args.threshold}_{args.interval}_objectness{args.objectness}re.txt")
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )