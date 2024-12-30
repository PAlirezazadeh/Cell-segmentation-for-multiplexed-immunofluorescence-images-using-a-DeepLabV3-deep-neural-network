#----------------------------------------------------#
#   Integrate single image prediction, video detection, and FPS testing
#   into one Python file, switching modes by specifying the mode variable.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   If you want to modify the colors corresponding to each class, go to the __init__ function and modify self.colors.
    #-------------------------------------------------------------------------#
    deeplab = DeeplabV3()
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the test mode:
    #   'predict'           Represents single image prediction. If you want to modify the prediction process, such as saving the image, cropping the object, etc., refer to the detailed comments below.
    #   'video'             Represents video detection, can use a webcam or video for detection, details are below.
    #   'fps'               Represents FPS testing, the image used is street.jpg inside the img folder, details below.
    #   'dir_predict'       Represents batch detection by traversing a folder and saving the results. By default, it traverses the img folder and saves to the img_out folder, details below.
    #   'export_onnx'       Represents exporting the model as ONNX, requires PyTorch version 1.7.1 or above.
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count               Specifies whether to perform pixel counting of the target (i.e., area) and ratio calculation.
    #   name_classes        Specifies the categories, similar to the ones in json_to_dataset, used for printing the categories and their quantities.
    #
    #   count and name_classes are only effective when mode='predict'.
    #-------------------------------------------------------------------------#
    count           = True
    name_classes    = ["background","cell"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Specifies the path of the video. When video_path=0, it indicates using the webcam.
    #                       If you want to detect a video, set it as video_path = "xxx.mp4", which represents loading the "xxx.mp4" file in the root directory.
    #   video_save_path     Specifies the path to save the video. When video_save_path="" it means the video will not be saved.
    #                       If you want to save the video, set it as video_save_path = "yyy.mp4", which represents saving it as "yyy.mp4" in the root directory.
    #   video_fps           Specifies the fps for the saved video.
    #
    #   video_path, video_save_path, and video_fps are only effective when mode='video'.
    #   When saving the video, you need to either press ctrl+c to exit or let the video run to the last frame for the saving process to complete.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Specifies the number of image detections when measuring fps. In theory, the larger the test_interval, the more accurate the fps measurement.
    #   fps_image_path      Specifies the image path used for fps testing.
    #   
    #   test_interval and fps_image_path are only effective when mode='fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Specifies the folder path of images used for detection.
    #   dir_save_path       Specifies the folder path where detected images will be saved.
    #   
    #   dir_origin_path and dir_save_path are only effective when mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify            Use Simplify for ONNX optimization.
    #   onnx_save_path      Specifies the path to save the ONNX model.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/deeplabV3-models.onnx"

    if mode == "predict":
        '''
        There are a few important points in predict.py:
        1. This code does not directly support batch prediction. If you want to perform batch predictions, 
           you can use os.listdir() to iterate through a folder and use Image.open to open image files for prediction.
           You can refer to get_miou_prediction.py for the detailed process, where batch iteration is implemented.
        2. To save the predicted image, you can use r_image.save("img.jpg") to save it.
        3. If you do not want to blend the original image and the segmentation result, set the blend parameter to False.
        4. If you want to extract specific areas based on the mask, you can refer to the drawing part in the detect_image 
           function. By evaluating the predicted result, you can determine the class of each pixel and extract the 
           corresponding part based on its class.
        
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*(self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*(self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*(self.colors[c][2])).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to correctly read the camera (video). Please check if the camera is properly installed (or if the video path is correctly specified).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert format from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # Perform detection
            frame = np.array(deeplab.detect_image(frame))
            # Convert format from RGB to BGR for OpenCV display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
