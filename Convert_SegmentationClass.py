#--------------------------------------------------------#
#   This script adjusts the format of the labels (segmentation masks).
#--------------------------------------------------------#
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#-----------------------------------------------------------------------------------#
#   Origin_SegmentationClass_path   The path where the original labels are located.
#   Out_SegmentationClass_path      The path where the output labels are located.
#                                   The processed labels are grayscale images; if the value set is too small, the details may not be visible.
#-----------------------------------------------------------------------------------#
Origin_SegmentationClass_path   = "patch-mask"
Out_SegmentationClass_path      = "SegmentationClass"

#-----------------------------------------------------------------------------------#
#   Origin_Point_Value  The pixel values corresponding to the original labels.
#   Out_Point_Value     The pixel values corresponding to the output labels.
#                       Origin_Point_Value needs to correspond one-to-one with Out_Point_Value.
#   For example, when:
#   Origin_Point_Value = np.array([0, 255])；Out_Point_Value = np.array([0, 1])
#   It means adjusting the pixel points with a value of 0 in the original labels to 0, and the pixel points with a value of 255 in the original labels to 1.
#-----------------------------------------------------------------------------------#
Origin_Point_Value              = np.array([0, 255])
Out_Point_Value                 = np.array([0, 1])

if __name__ == "__main__":
    if not os.path.exists(Out_SegmentationClass_path):
        os.makedirs(Out_SegmentationClass_path)

    png_names = os.listdir(Origin_SegmentationClass_path)
    for png_name in tqdm(png_names):
        png     = Image.open(os.path.join(Origin_SegmentationClass_path, png_name))
        w, h    = png.size
        
        png     = np.array(png)
        out_png = np.zeros([h, w])
        for i in range(len(Origin_Point_Value)):
            mask = png[:, :] == Origin_Point_Value[i]
            if len(np.shape(mask)) > 2:
                mask = mask.all(-1)
            out_png[mask] = Out_Point_Value[i]
        
        out_png = Image.fromarray(np.array(out_png, np.uint8))
        out_png.save(os.path.join(Out_SegmentationClass_path, png_name))
