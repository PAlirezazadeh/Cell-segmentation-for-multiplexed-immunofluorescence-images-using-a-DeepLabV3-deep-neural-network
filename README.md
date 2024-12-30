## Cell segmentation for multiplexed immunofluorescence images using a DeepLabV3+ deep neural network

---

### Environment
conda create -n MELC python=3.9  
conda activate MELC  
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch  
pip install -r requirements.txt  

### Notes
The deeplab_mobilenetv2.pth and deeplab_xception.pth files in the code are trained on the VOC extended dataset. Remember to adjust the backbone when training and predicting.  

### Training Steps
1. This document uses VOC format for training.
2. Pre-processing images and labels by applying Convert_JPEGImages.py and Convert_SegmentationClass.py.
3. Place label files in the SegmentationClass folder under VOC2007 in VOCdevkit.
4. Place image files in the JPEGImages folder under VOC2007 in VOCdevkit.
5. Generate corresponding txt files using voc_annotation.py before training.
6. In the train.py folder, select the backbone model and downsampling factor you want to use. The backbone models provided are mobilenet and xception. Choose between downsampling factors of 8 and 16. Note that the pretrained model needs to match the backbone model.
7. Modify num_classes in train.py to the number of classes +1.
8. Run train.py to start training.

### Prediction Steps
1. Train according to the training steps.
2. In deeplab.py, modify model_path, num_classes, and backbone to match the trained files. model_path corresponds to the weight file in the logs folder, num_classes is the number of classes to predict +1, and backbone is the backbone feature extraction network used.

_defaults = {  
    #----------------------------------------#  
    #   model_path points to the weight file in the logs folder  
    #----------------------------------------#  
    "model_path"        : 'model_data/deeplab_mobilenetv2.pth',  
    #----------------------------------------#  
    #   Number of classes to differentiate +1  
    #----------------------------------------#  
    "num_classes"       : 2,  
    #----------------------------------------#  
    #   Backbone network used  
    #----------------------------------------#  
    "backbone"          : "mobilenet",  
    #----------------------------------------#  
    #   Input image size  
    #----------------------------------------#  
    "input_shape"       : [512, 512],  
    #----------------------------------------#  
    #   Downsampling factor, generally 8 or 16  
    #   Must match the training setting  
    #----------------------------------------#  
    "downsample_factor" : 16,  
    #--------------------------------#  
    #   blend parameter controls whether  
    #   to blend the recognition result with the original image  
    #--------------------------------#  
    "blend"             : True,  
    #-------------------------------#  
    #   Whether to use Cuda  
    #   Set to False if no GPU is available  
    #-------------------------------#  
    "cuda"              : True,  
}  

3. Run predict.py, input:
img/image_20220321_FOV3.jpg
to complete the prediction.
4. In predict.py, you can set up FPS testing, testing an entire folder, and video detection.

### Evaluation Steps
Set num_classes in get_miou.py to the number of predicted classes +1.
Set name_classes in get_miou.py to the categories to be distinguished.
Run get_miou.py to obtain the mIOU score.












