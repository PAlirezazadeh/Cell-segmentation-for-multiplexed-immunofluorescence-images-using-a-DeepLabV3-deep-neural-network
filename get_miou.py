import os

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
When performing metric evaluation, please note the following points:
1. The generated images are grayscale. Since the values are relatively small, they may not be visible in PNG format, so it is normal to see an almost completely black image.
2. This file computes the mIoU and all metrics for the validation set.
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode specifies what the file calculates during execution.
    #   miou_mode = 0 means the entire mIoU calculation process, including obtaining prediction results and calculating mIoU.
    #   miou_mode = 1 means only obtaining prediction results.
    #   miou_mode = 2 means only calculating mIoU.
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   Number of classes + 1, e.g., 1 + 1
    #------------------------------#
    num_classes     = 2
    #--------------------------------------------#
    #   Categories to distinguish, same as in json_to_dataset
    #--------------------------------------------#
    name_classes    = ["background","cell"]
    #-------------------------------------------------------#
    #   Path to the VOC dataset folder
    #   Defaults to the VOC dataset in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # Execute the function to calculate mIoU
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
