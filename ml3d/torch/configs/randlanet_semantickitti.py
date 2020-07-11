from os.path import join

# model settings
model       = dict(
    k_n             = 16,  # KNN,
    num_layers      = 4,  # Number of layers
    num_points      = 4096 * 11,  # Number of input points
    num_classes     = 19,  # Number of valid classes

    sub_grid_size   = 0.06,  # preprocess_parameter
    sub_sampling_ratio = [4, 4, 4, 4],
    num_sub_points = [4096 * 11 // 4, 4096 * 11 // 16, 4096 * 11 // 64, 4096 * 11 // 256],

    d_in            = 3,
    d_feature       = 8,
    d_out           = [16, 64, 128, 256] ,

    ckpt_path       = './ml3d/torch/checkpoint/randlanet_semantickitti.pth'
)

pipeline    = dict(
    batch_size          = 2,
    val_batch_size      = 2,
    test_batch_size     = 3,
    max_epoch           = 100,  # maximum epoch during training
    learning_rate       = 1e-2,  # initial learning rate
    lr_decays           = {i: 0.95 for i in range(0, 500)},
    save_ckpt_freq      = 20,
    adam_lr             = 1e-2,
    scheduler_gamma     = 0.95,


    # logs
    main_log_dir        = './logs',
    model_name          = 'RandLANet',
    train_sum_dir       = 'train_log',
    )

dataset = dict(
    dataset_path        = '/home/yiling/d2T/intel2020/datasets/semanticKITTI/data_odometry_velodyne/dataset/sequences_0.06',
    test_result_folder  = './test',

    training_split      = ['00', '01', '02', '03', '04', '05', 
                            '06', '07', '09', '10'],
    validation_split    = ['08'],
    test_split_number   = 11,
    class_weights       = [55437630, 320797, 541736, 2578735, 3274484, 552662, 
                        184064, 78858, 240942562, 17294618, 170599734, 
                        6369672, 230413074, 101130274, 476491114,9833174, 
                        129609852, 4506626, 1168181],
    )

