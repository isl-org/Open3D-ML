from os.path import join
import numpy as np


# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # Dataset name
    dataset = ''

    # Type of network model
    dataset_task = ''

    # Number of classes in the dataset
    num_classes = 0

    # Dimension of input points
    in_points_dim = 3

    # Dimension of input features
    in_features_dim = 1

    # Radius of the input sphere (ignored for models, only used for point clouds)
    in_radius = 1.0

    # Number of CPU threads for the input pipeline
    input_threads = 8

    ##################
    # Model parameters
    ##################

    # Architecture definition. List of blocks
    architecture = []

    # Decide the mode of equivariance and invariance
    equivar_mode = ''
    invar_mode = ''

    # Dimension of the first feature maps
    first_features_dim = 64

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.99

    # For segmentation models : ratio between the segmented area and the input area
    segmentation_ratio = 1.0

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Kernel point influence radius
    KP_extent = 1.0

    # Influence function when d < KP_extent. ('constant', 'linear', 'gaussian') When d > KP_extent, always zero
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    # Decide if you sum all kernel point influences, or if you only take the influence of the closest KP
    aggregation_mode = 'sum'

    # Fixed points in the kernel : 'none', 'center' or 'verticals'
    fixed_kernel_points = 'center'

    # Use modulateion in deformable convolutions
    modulated = False

    # For SLAM datasets like SemanticKitti number of frames used (minimum one)
    n_frames = 1

    # For SLAM datasets like SemanticKitti max number of point in input cloud + validation
    max_in_points = 0
    val_radius = 51.0
    max_val_points = 50000

    #####################
    # Training parameters
    #####################

    # Network optimizer parameters (learning rate and momentum)
    learning_rate = 1e-3
    momentum = 0.9

    # Learning rate decays. Dictionary of all decay values with their epoch {epoch: decay}.
    lr_decays = {200: 0.2, 300: 0.2}

    # Gradient clipping value (negative means no clipping)
    grad_clip_norm = 100.0

    # Augmentation parameters
    augment_scale_anisotropic = True
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_symmetries = [False, False, False]
    augment_rotation = 'vertical'
    augment_noise = 0.005
    augment_color = 0.7

    # Augment with occlusions (not implemented yet)
    augment_occlusion = 'none'
    augment_occlusion_ratio = 0.2
    augment_occlusion_num = 1

    # Regularization loss importance
    weight_decay = 1e-3

    # The way we balance segmentation loss DEPRECATED
    segloss_balance = 'none'

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    class_w = []

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0  # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.0  # Distance of repulsion for deformed kernel points

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Maximal number of epochs
    max_epoch = 1000

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 100

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Do we nee to save convergence
    saving = True
    saving_path = None

    def __init__(self):
        """
        Class Initialyser
        """

        # Number of layers
        self.num_layers = len([
            block for block in self.architecture
            if 'pool' in block or 'strided' in block
        ]) + 1

        ###################
        # Deform layer list
        ###################
        #
        # List of boolean indicating which layer has a deformable convolution
        #

        layer_blocks = []
        self.deform_layers = []
        arch = self.architecture
        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block
                    or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    deform_layer = True

            if 'pool' in block or 'strided' in block:
                if 'deformable' in block:
                    deform_layer = True

            self.deform_layers += [deform_layer]
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

    def load(self, path):

        filename = join(path, 'parameters.txt')
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Class variable dictionary
        for line in lines:
            line_info = line.split()
            if len(line_info) > 2 and line_info[0] != '#':

                if line_info[2] == 'None':
                    setattr(self, line_info[0], None)

                elif line_info[0] == 'lr_decay_epochs':
                    self.lr_decays = {
                        int(b.split(':')[0]): float(b.split(':')[1])
                        for b in line_info[2:]
                    }

                elif line_info[0] == 'architecture':
                    self.architecture = [b for b in line_info[2:]]

                elif line_info[0] == 'augment_symmetries':
                    self.augment_symmetries = [
                        bool(int(b)) for b in line_info[2:]
                    ]

                elif line_info[0] == 'num_classes':
                    if len(line_info) > 3:
                        self.num_classes = [int(c) for c in line_info[2:]]
                    else:
                        self.num_classes = int(line_info[2])

                elif line_info[0] == 'class_w':
                    self.class_w = [float(w) for w in line_info[2:]]

                elif hasattr(self, line_info[0]):
                    attr_type = type(getattr(self, line_info[0]))
                    if attr_type == bool:
                        setattr(self, line_info[0],
                                attr_type(int(line_info[2])))
                    else:
                        setattr(self, line_info[0], attr_type(line_info[2]))

        self.saving = True
        self.saving_path = path
        self.__init__()

    def save(self):

        with open(join(self.saving_path, 'parameters.txt'), "w") as text_file:

            text_file.write('# -----------------------------------#\n')
            text_file.write('# Parameters of the training session #\n')
            text_file.write('# -----------------------------------#\n\n')

            # Input parameters
            text_file.write('# Input parameters\n')
            text_file.write('# ****************\n\n')
            text_file.write('dataset = {:s}\n'.format(self.dataset))
            text_file.write('dataset_task = {:s}\n'.format(self.dataset_task))
            if type(self.num_classes) is list:
                text_file.write('num_classes =')
                for n in self.num_classes:
                    text_file.write(' {:d}'.format(n))
                text_file.write('\n')
            else:
                text_file.write('num_classes = {:d}\n'.format(
                    self.num_classes))
            text_file.write('in_points_dim = {:d}\n'.format(
                self.in_points_dim))
            text_file.write('in_features_dim = {:d}\n'.format(
                self.in_features_dim))
            text_file.write('in_radius = {:.6f}\n'.format(self.in_radius))
            text_file.write('input_threads = {:d}\n\n'.format(
                self.input_threads))

            # Model parameters
            text_file.write('# Model parameters\n')
            text_file.write('# ****************\n\n')

            text_file.write('architecture =')
            for a in self.architecture:
                text_file.write(' {:s}'.format(a))
            text_file.write('\n')
            text_file.write('equivar_mode = {:s}\n'.format(self.equivar_mode))
            text_file.write('invar_mode = {:s}\n'.format(self.invar_mode))
            text_file.write('num_layers = {:d}\n'.format(self.num_layers))
            text_file.write('first_features_dim = {:d}\n'.format(
                self.first_features_dim))
            text_file.write('use_batch_norm = {:d}\n'.format(
                int(self.use_batch_norm)))
            text_file.write('batch_norm_momentum = {:.6f}\n\n'.format(
                self.batch_norm_momentum))
            text_file.write('segmentation_ratio = {:.6f}\n\n'.format(
                self.segmentation_ratio))

            # KPConv parameters
            text_file.write('# KPConv parameters\n')
            text_file.write('# *****************\n\n')

            text_file.write('first_subsampling_dl = {:.6f}\n'.format(
                self.first_subsampling_dl))
            text_file.write('num_kernel_points = {:d}\n'.format(
                self.num_kernel_points))
            text_file.write('conv_radius = {:.6f}\n'.format(self.conv_radius))
            text_file.write('deform_radius = {:.6f}\n'.format(
                self.deform_radius))
            text_file.write('fixed_kernel_points = {:s}\n'.format(
                self.fixed_kernel_points))
            text_file.write('KP_extent = {:.6f}\n'.format(self.KP_extent))
            text_file.write('KP_influence = {:s}\n'.format(self.KP_influence))
            text_file.write('aggregation_mode = {:s}\n'.format(
                self.aggregation_mode))
            text_file.write('modulated = {:d}\n'.format(int(self.modulated)))
            text_file.write('n_frames = {:d}\n'.format(self.n_frames))
            text_file.write('max_in_points = {:d}\n\n'.format(
                self.max_in_points))
            text_file.write('max_val_points = {:d}\n\n'.format(
                self.max_val_points))
            text_file.write('val_radius = {:.6f}\n\n'.format(self.val_radius))

            # Training parameters
            text_file.write('# Training parameters\n')
            text_file.write('# *******************\n\n')

            text_file.write('learning_rate = {:f}\n'.format(
                self.learning_rate))
            text_file.write('momentum = {:f}\n'.format(self.momentum))
            text_file.write('lr_decay_epochs =')
            for e, d in self.lr_decays.items():
                text_file.write(' {:d}:{:f}'.format(e, d))
            text_file.write('\n')
            text_file.write('grad_clip_norm = {:f}\n\n'.format(
                self.grad_clip_norm))

            text_file.write('augment_symmetries =')
            for a in self.augment_symmetries:
                text_file.write(' {:d}'.format(int(a)))
            text_file.write('\n')
            text_file.write('augment_rotation = {:s}\n'.format(
                self.augment_rotation))
            text_file.write('augment_noise = {:f}\n'.format(
                self.augment_noise))
            text_file.write('augment_occlusion = {:s}\n'.format(
                self.augment_occlusion))
            text_file.write('augment_occlusion_ratio = {:.6f}\n'.format(
                self.augment_occlusion_ratio))
            text_file.write('augment_occlusion_num = {:d}\n'.format(
                self.augment_occlusion_num))
            text_file.write('augment_scale_anisotropic = {:d}\n'.format(
                int(self.augment_scale_anisotropic)))
            text_file.write('augment_scale_min = {:.6f}\n'.format(
                self.augment_scale_min))
            text_file.write('augment_scale_max = {:.6f}\n'.format(
                self.augment_scale_max))
            text_file.write('augment_color = {:.6f}\n\n'.format(
                self.augment_color))

            text_file.write('weight_decay = {:f}\n'.format(self.weight_decay))
            text_file.write('segloss_balance = {:s}\n'.format(
                self.segloss_balance))
            text_file.write('class_w =')
            for a in self.class_w:
                text_file.write(' {:.6f}'.format(a))
            text_file.write('\n')
            text_file.write('deform_fitting_mode = {:s}\n'.format(
                self.deform_fitting_mode))
            text_file.write('deform_fitting_power = {:.6f}\n'.format(
                self.deform_fitting_power))
            text_file.write('deform_lr_factor = {:.6f}\n'.format(
                self.deform_lr_factor))
            text_file.write('repulse_extent = {:.6f}\n'.format(
                self.repulse_extent))
            text_file.write('batch_num = {:d}\n'.format(self.batch_num))
            text_file.write('val_batch_num = {:d}\n'.format(
                self.val_batch_num))
            text_file.write('max_epoch = {:d}\n'.format(self.max_epoch))
            if self.epoch_steps is None:
                text_file.write('epoch_steps = None\n')
            else:
                text_file.write('epoch_steps = {:d}\n'.format(
                    self.epoch_steps))
            text_file.write('validation_size = {:d}\n'.format(
                self.validation_size))
            text_file.write('checkpoint_gap = {:d}\n'.format(
                self.checkpoint_gap))
