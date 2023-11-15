import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        f = open("results.txt" , "w")
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            if outputs[0].get("instances").get("scores").size()[0] != 0:
              before_pcb = outputs[0].get("instances").get("scores")[0].item()
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
                if outputs[0].get("instances").get("scores").size()[0] != 0:
                  after_pcb = outputs[0].get("instances").get("scores")[0].item()
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)
            
            instances = outputs[0].get("instances")
            
            number_of = instances.get("scores").size(0) 

            count = 0
            x1,y1,x2,y2,prob = 0,0,0,0,0
            wrong_count = 0
            if count == number_of :
              f.write("{},0,0,0,0,0,0 \n".format(idx + 1))
            while(count < number_of ):
                if(True): #instances.get("pred_classes")[count].item() == -1 or instances.get("pred_classes")[count].item() == -1
                  wrong_count = wrong_count + 1
                  
                  x1 = int(instances.get("pred_boxes").tensor[count][0].item())
                  y1 = int(instances.get("pred_boxes").tensor[count][1].item())
                  x2 = int(instances.get("pred_boxes").tensor[count][2].item())
                  y2 = int(instances.get("pred_boxes").tensor[count][3].item())
                  if(instances.get("scores").size(0) == 1):
                      prob = float(instances.get("scores").item())
                  else:
                      prob = float(instances.get("scores").data[count].item()) 
                  f.write("{},{},{},{},{},{},{}".format(idx + 1, x1,y1,x2-x1,y2-y1, prob, instances.get("pred_classes")[count].item() ) + "\n")
                count = count + 1
                if(count - wrong_count == number_of):
                  print("0lı kare")
                  f.write("{},0,0,0,0,0,0 \n".format(idx + 1))
            
            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def inference_on_dataset_nopcb(model, data_loader, evaluator, evaluator2, cfg=None):

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()
    evaluator2.reset()
    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        f = open("results.txt" , "w")
        f_pcb = open("results_pcb.txt" , "w")
        f.close()
        f_pcb.close()
        f = open("results.txt" , "a") 
        f_pcb = open("results_pcb.txt", "a")  
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            outputs1 = model(inputs)
            if outputs[0].get("instances").get("scores").size()[0] != 0:
              before_pcb = outputs[0].get("instances")
            if cfg.TEST.PCB_ENABLE:
                outputs = model(inputs)
                outputs = pcb.execute_calibration(inputs, outputs)
                if outputs[0].get("instances").get("scores").size()[0] != 0:
                  after_pcb = outputs[0].get("instances")
            if outputs[0].get("instances").get("scores").size()[0] == 0:
                before_pcb = outputs[0].get("instances")
                after_pcb = before_pcb
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)
            evaluator2.process(inputs,outputs1)
            #print(outputs==outputs1)
            instances = outputs[0].get("instances")
            
            number_of = instances.get("scores").size(0) 

          ################################################################
            ##before_pcb için dosya yazdırma
            #number_of = before_pcb.get("scores").size(0) 
            count = 0
            wrong_count = 0
            if count == number_of :
              f.write("{},0,0,0,0,0,0 \n".format(idx))
            while(count < number_of ):

                if(before_pcb.get("pred_classes")[count].item() == 0): 
                  wrong_count = wrong_count + 1
                  x1 = int(before_pcb.get("pred_boxes").tensor[count][0].item())
                  y1 = int(before_pcb.get("pred_boxes").tensor[count][1].item())
                  x2 = int(before_pcb.get("pred_boxes").tensor[count][2].item())
                  y2 = int(before_pcb.get("pred_boxes").tensor[count][3].item())
                  if(before_pcb.get("scores").size(0) == 1):
                      prob = float(before_pcb.get("scores").item())
                  else:
                      prob = float(before_pcb.get("scores").data[count].item()) 
                  f.write("{},{},{},{},{},{},{}".format(idx + 1, x1,y1,x2-x1,y2-y1, prob, before_pcb.get("pred_classes")[count].item() ) + "\n")
                count = count + 1
                if(count - wrong_count == number_of):
                  #print("0lı kare")
                  f.write("{},0,0,0,0,0,0 \n".format(idx + 1))
            
            #after_pcb için dosya yazdırma
            number_of = after_pcb.get("scores").size(0) 
            count = 0
            x1,y1,x2,y2,prob_2 = 0,0,0,0,0
            wrong_count = 0
            if count == number_of :
              #sim.write("{},0 \n".format(idx))
              f_pcb.write("{},0,0,0,0,0,0 \n".format(idx + 1))
            while(count < number_of ):

                if(after_pcb.get("pred_classes")[count].item() == 0): 
                  wrong_count = wrong_count + 1
                  x1 = int(after_pcb.get("pred_boxes").tensor[count][0].item())
                  y1 = int(after_pcb.get("pred_boxes").tensor[count][1].item())
                  x2 = int(after_pcb.get("pred_boxes").tensor[count][2].item())
                  y2 = int(after_pcb.get("pred_boxes").tensor[count][3].item())
                  if(before_pcb.get("scores").size(0) == 1):
                      prob = float(before_pcb.get("scores").item())
                  else:
                      prob = float(before_pcb.get("scores").data[count].item()) 
                  if(after_pcb.get("scores").size(0) == 1):
                      prob_2 = float(after_pcb.get("scores").item())
                  else:
                      prob_2 = float(after_pcb.get("scores").data[count].item()) 
                  f_pcb.write("{},{},{},{},{},{},{}".format(idx + 1 , x1,y1,x2-x1,y2-y1, prob_2, after_pcb.get("pred_classes")[count].item() ) + "\n")
                  #sim.write("{},{} \n".format(idx,2*prob_2 - prob))
                count = count + 1
                if(count - wrong_count == number_of):
                  #print("0lı kare")
                  f_pcb.write("{},0,0,0,0,0,0 \n".format(idx + 1))    
                  #sim.write("{},0 \n".format(idx))  
            ##################################################################
            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    results2 = evaluator2.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, results2



@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
