# model settings
model       = dict(
    k_n             = 16,  # KNN,
    num_layers      = 5,  # Number of layers
    num_points      = 4096 * 10,  # Number of input points
    num_classes     = 13,  # Number of valid classes

    sub_grid_size   = 0.04,  # preprocess_parameter
    sub_sampling_ratio = [4, 4, 4, 4, 2],
    num_sub_points = [4096 * 10 // 4, 4096 * 10 // 16, 4096 * 10 // 64, 4096 * 10 // 256, 4096 * 10 // 512],

    d_in            = 6,
    d_feature       = 8,
    d_out           = [16, 64, 128, 256, 512],

    ckpt_path       = './ml3d/torch/checkpoint/randlanet_s3dis.pth'
)

pipeline    = dict(
    batch_size          = 2,
    val_batch_size      = 2,
    test_batch_size     = 3,
    max_epoch           = 100,  # maximum epoch during training
    learning_rate       = 1e-2,  # initial learning rate
    #lr_decays           = {0.95 for i in range(0, 500)},
    save_ckpt_freq      = 20,
    adam_lr             = 1e-2,
    scheduler_gamma     = 0.95,


    # logs
    main_log_dir        = './logs',
    model_name          = 'RandLANet',
    train_sum_dir       = 'train_log',
    )

dataset = dict(
    dataset_path    = '/Users/sanskara/Downloads/Stanford3dDataset_v1.2_Aligned_Version/',
    cache_path        = '/Users/sanskara/Downloads/Stanford3dDataset_v1.2_Aligned_Version/cache/',
    prepro_grid_size    = 0.04,
    num_points      = 4096 * 10,
    test_result_folder  = './test',

    test_area_idx = 3, # Area_0 to Area_6
    class_weights = [3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837],
    ignored_label_inds = []
    )
