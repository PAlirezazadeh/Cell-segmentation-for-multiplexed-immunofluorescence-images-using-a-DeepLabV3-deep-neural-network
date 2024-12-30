import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   To increase the test set, modify trainval_percent
#   Modify train_percent to change the proportion of the validation set to 9:1
#   
#   Currently, this library uses the test set as the validation set, without a separate test set
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9
#-------------------------------------------------------#
#   Points to the folder where the VOC dataset is located
#   Defaults to the VOC dataset in the root directory
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("Checking if the dataset format meets the requirements; this may take some time.")
    classes_nums        = np.zeros([256], np.int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label image %s not found. Please check if the file exists at the specified path and ensure the extension is png." % (png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The shape of the label image %s is %s, which is not a grayscale image or 8-bit color image. Please check the dataset format carefully." % (name, str(np.shape(png))))
            print("The label image needs to be either a grayscale or 8-bit color image, where the value of each pixel indicates the class of that pixel." % (name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("Print the pixel values and their counts.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("Detected that the pixel values in the labels only include 0 and 255. The data format is incorrect.")
        print("For binary classification, the labels should have pixel values of 0 for the background and 1 for the object.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("Detected that the labels only contain background pixels. The data format is incorrect. Please carefully check the dataset format.")

    print("Images in JPEGImages should be .jpg files, and images in SegmentationClass should be .png files.")
    print("If the format is incorrect, refer to part 2 of Training Steps.")
