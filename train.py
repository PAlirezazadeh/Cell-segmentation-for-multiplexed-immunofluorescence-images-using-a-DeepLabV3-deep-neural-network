"""
Created on Tue Mar 26 10:52:31 2024

@author: palirezazadeh
"""

import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
When training your own semantic segmentation model, make sure to pay attention to the following points:

1) Check the format before training to ensure it meets the requirements. The library requires the dataset to be in VOC format. The prepared contents should include input images and labels:

- Input images should be in .jpg format and do not need to be of a fixed size, as they will be automatically resized before being fed into the model.

- Grayscale images will automatically be converted to RGB for training, so no manual changes are necessary.

- If the input images are not in .jpg format, you'll need to batch-convert them to .jpg before starting the training.

- Labels should be .png images, and they also don’t need to be of a fixed size, as they will be automatically resized before being fed into the model.

2) Loss value is used to judge whether the model has converged. The important thing is to observe a convergence trend, meaning the validation set loss should continually decrease. If the validation set loss remains largely unchanged, the model has likely converged.

- The specific size of the loss value doesn’t have much meaning; whether the value is large or small depends on the way the loss is calculated. It’s not necessarily better if the loss approaches zero. If you want the loss value to look smaller, you can directly divide it by 10,000 in the corresponding loss function.
- The loss values during training will be saved in the logs folder, under a folder named loss_%Y_%m_%d_%H_%M_%S.

3) The trained weight files are saved in the logs folder. Each training epoch consists of several steps, and during each step, gradient descent is performed once.

If you only train for a few steps, the model won’t be saved. Be sure to understand the concepts of epochs and steps clearly.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use Cuda
    #           If no GPU is available, set this to False
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    Used to set a fixed random seed
    #           This ensures that each independent training run produces the same results
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use multi-GPU distributed training on a single machine
    #   Terminal commands are only supported on Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the GPUs in Ubuntu.
    #   On Windows systems, DP mode is used by default to call all GPUs, and DDP is not supported.
    
    # **DP Mode:**
    # Set `distributed = False`
    # In the terminal, enter: `CUDA_VISIBLE_DEVICES=0,1 python train.py`

    # **DDP Mode:**
    # Set `distributed = True`
    # In the terminal, enter: `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py`"

    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use synchronized batch normalization, available in DDP mode with multiple GPUs
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #   This can reduce memory usage by about half, and requires PyTorch 1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = True
    #-----------------------------------------------------#
    #   num_classes     This must be modified when training on your own dataset
    #   Set it to the number of classes you need + 1, e.g., 1+1
    #-----------------------------------------------------#
    num_classes     = 2
    #---------------------------------#
    #   Backbone network used:
    #   mobilenet
    #   xception
    #---------------------------------#
    backbone        = "mobilenet"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use the pretrained weights of the backbone network. The backbone weights are loaded when building the model.
    #   - If model_path is set, the backbone weights don't need to be loaded, and the value of pretrained is irrelevant.
    #   - If model_path is not set and pretrained = True, only the backbone is loaded to start training.
    #   - If model_path is not set and pretrained = False and Freeze_Train = False, training starts from scratch with no freezing of the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Please refer to the README for downloading the weight files; they can be downloaded via a cloud drive. 
    #   The pretrained weights of the model are general across different datasets because the features are general.
    #   The important part of the pretrained weights is the weights of the backbone feature extraction network, which are used for feature extraction.
    #   Pretrained weights are necessary in 99% of cases; without them, the backbone weights would be too random, feature extraction would be poor, and the training results would not be good.
    #   When training your own dataset, it's normal to see a dimension mismatch, as the predictions are different and thus the dimensions naturally don't match.
    #
    #   If training is interrupted, you can set `model_path` to the weight file in the logs folder and reload the partially trained weights.
    #   Also, modify the parameters for the frozen or unfrozen stages below to ensure continuity in model training epochs.
    #   
    #   When `model_path = ''`, no weights from the entire model are loaded.
    #
    #   This uses the weights of the entire model, so they are loaded in `train.py`; the `pretrain` setting does not affect this weight loading.
    #   If you want the model to start training from the backbone's pretrained weights, set `model_path = ''` and `pretrain = True`, which will load only the backbone.
    #   If you want the model to train from scratch, set `model_path = ''`, `pretrain = False`, and `Freeze_Train = False`, which will start training from scratch with no backbone freezing.
    #   
    #   Generally speaking, training the network from scratch gives poor results because the weights are too random and feature extraction is ineffective, so it is highly, highly, highly discouraged to train from scratch!
    #   If you must train from scratch, you should first explore the ImageNet dataset, train a classification model to obtain the backbone weights, as the backbone of the classification model and this model are shared. You can then train based on that.

    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/deeplab_mobilenetv2.pth"
    #---------------------------------------------------------#
    #   downsample_factor   Downsampling factor, 8 or 16
    #                       A factor of 8 has less downsampling, theoretically yielding better results.
    #                       However, it also requires more GPU memory.

    #---------------------------------------------------------#
    downsample_factor   = 8
    #------------------------------#
    #   Input image size
    #------------------------------#
    input_shape         = [512, 512]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two phases: the freezing phase and the unfreezing phase. The freezing phase is designed to accommodate those with limited machine performance.
    #   Freezing training requires less GPU memory; if the GPU is very poor, you can set `Freeze_Epoch` equal to `UnFreeze_Epoch` to only perform freezing training.
    #   
    #   Here are some suggested parameter settings. Adjust according to your needs:
    #   (1) Training from the pretrained weights of the entire model:
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Not Frozen)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Not Frozen)
    #       Note: `UnFreeze_Epoch` can be adjusted between 100-300.
    #   (2) Training from the pretrained weights of the backbone network:
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Not Frozen)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 120, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 120, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Not Frozen)
    #       Note: When training from the pretrained backbone weights, the backbone weights may not be ideal for semantic segmentation and require more training to escape local minima.
    #             `UnFreeze_Epoch` can be adjusted between 120-300.
    #             Adam generally converges faster than SGD, so `UnFreeze_Epoch` can theoretically be smaller, but more epochs are still recommended.
    #   (3) Batch size settings:
    #       Within the GPU's acceptable range, larger is better. Memory issues are unrelated to dataset size; if you encounter out-of-memory (OOM or CUDA out of memory) errors, reduce the `batch_size`.
    #       Due to the influence of BatchNorm layers, the minimum batch size is 2, not 1.
    #       Normally, `Freeze_batch_size` should be 1-2 times the `Unfreeze_batch_size`. A large discrepancy is not recommended as it affects automatic learning rate adjustments.

    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freezing phase training parameters
    #   At this stage, the model's backbone is frozen, and the feature extraction network remains unchanged.
    #   It requires less GPU memory and only fine-tunes the network.
    #   Init_Epoch          The current training epoch when the model starts, which can be greater than `Freeze_Epoch`. For example:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #                       This setup will skip the freezing phase, starting directly from epoch 60 and adjusting the corresponding learning rate.
    #                       (Used for checkpoint continuation)
    #   Freeze_Epoch        Epochs for freezing the model during training
    #                       (Ineffective when `Freeze_Train=False`)
    #   Freeze_batch_size   Batch size for freezing the model during training
    #                       (Ineffective when `Freeze_Train=False`)

    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    #   Unfreezing phase training parameters
    #   At this stage, the model's backbone is no longer frozen, and the feature extraction network will be updated.
    #   It requires more GPU memory, and all network parameters will change.
    #   UnFreeze_Epoch          Total number of epochs for training the model
    #   Unfreeze_batch_size     Batch size for the model after unfreezing

    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freezing training
    #                   By default, the backbone is first frozen for training and then unfrozen for further training.

    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, and learning rate decay

    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate for the model
    #                   For the Adam optimizer, it is recommended to set Init_lr=5e-4
    #                   For the SGD optimizer, it is recommended to set Init_lr=7e-3
    #   Min_lr          The minimum learning rate for the model, defaulting to 0.01 of the maximum learning rate

    #------------------------------------------------------------------#
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Type of optimizer to use; options include adam, sgd
    #                   For the Adam optimizer, it is recommended to set Init_lr=5e-4
    #                   For the SGD optimizer, it is recommended to set Init_lr=7e-3
    #   momentum        Momentum parameter used within the optimizer
    #   weight_decay    Weight decay to prevent overfitting
    #                   Adam may cause issues with weight decay; it is recommended to set weight_decay to 0 when using Adam.

    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay method used; options include 'step' and 'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     Number of epochs between saving weights
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        Folder where weights and log files are saved
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether to perform evaluation during training, with the evaluation target being the validation set
    #   eval_period     Number of epochs between evaluations; frequent evaluations are not recommended
    #                   Evaluations consume significant time, and frequent evaluations can make training very slow
    #   The mAP obtained here may differ from that obtained with `get_map.py` for two reasons:
    #   (1) The mAP obtained here is for the validation set.
    #   (2) The evaluation parameters set here are more conservative, aiming to speed up the evaluation process.

    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5

    #------------------------------------------------------------------#
    #   VOCdevkit_path  Path to the dataset
    #------------------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#
    #   Suggested options:
    #   When there are few classes (a few categories), set to True
    #   When there are many classes (dozens of categories), and the batch_size is relatively large (10 or more), set to True
    #   When there are many classes (dozens of categories), and the batch_size is relatively small (fewer than 10), set to False

    #------------------------------------------------------------------#
    dice_loss       = True
    #------------------------------------------------------------------#
    #   Whether to use focal loss to address class imbalance between positive and negative samples
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   Whether to assign different loss weights to different classes; by default, the weights are balanced.
    #   If set, make sure to provide it in numpy array format with a length equal to `num_classes`.
    #   For example:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)

    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     Sets whether to use multiple threads for data loading; 1 means disable multithreading
    #                   Enabling it will speed up data loading but will use more memory
    #                   In some cases, enabling multithreading in Keras can actually slow things down
    #                   Enable multithreading only when I/O is the bottleneck, i.e., when GPU computation speed is much faster than image loading speed.

    #------------------------------------------------------------------#
    num_workers         = 2

    seed_everything(seed)
    #------------------------------------------------------#
    #   Set the GPUs to be used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   Download pretrained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   For weight files, please refer to the README; download from Baidu Netdisk
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load according to the keys of the pretrained weights and the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Display keys that do not match
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mFriendly reminder: It is normal for the head section to not load, but an error if the Backbone section does not load.\033[0m")

    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    #   Torch 1.2 does not support AMP; it is recommended to use Torch 1.7.1 or above to correctly use fp16
    #   Therefore, Torch 1.2 will show "could not be resolved" here
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-GPU synchronized Batch Normalization (Bn)
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel operation
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   Read the corresponding txt file for the dataset
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   Total training epochs refer to the total number of times the entire dataset is traversed
        #   Total training steps refer to the total number of gradient descent steps
        #   Each training epoch contains several training steps, with each step performing one gradient descent.
        #   Here, only the minimum number of training epochs is suggested; there is no upper limit, and only the unfreezing part is considered in calculations
        #----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small to train. Please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] \033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] %d，Unfreeze_batch_size为%d，\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] %d。\033[0m"%(total_step, wanted_step, wanted_epoch))

        
    #------------------------------------------------------#
    #   The backbone feature extraction network features are generic; frozen training can speed up training
    #   It can also prevent weights from being damaged in the early stages of training.
    #   Init_Epoch is the starting epoch
    #   Interval_Epoch is the epoch for frozen training
    #   Epoch is the total number of training epochs
    #   If you encounter OOM or insufficient memory, please reduce the Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze a certain part of the training
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If not using frozen training, set batch_size directly to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size and adjust the learning rate adaptively
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Choose optimizer based on optimizer_type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   Get the formula for learning rate decay
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Determine the length of each epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   Record the mAP curve for evaluation
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has frozen training parts
            #   Unfreeze them and set the parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size and adjust the learning rate adaptively
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Obtain the formula for learning rate decay
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
