from phiseg.model_zoo import likelihoods, posteriors, priors
import tensorflow as tf
from tfwrapper import normalisation as tfnorm

experiment_name = 'phiseg_7_5_1annot'
log_dir_name = 'lidc2'

# architecture
posterior = posteriors.phiseg
likelihood = likelihoods.phiseg
prior = priors.phiseg
layer_norm = tfnorm.batch_norm
use_logistic_transform = False

latent_levels = 5
resolution_levels = 7
n0 = 32
zdim0 = 2
max_channel_power = 4  # max number of channels will be n0*2**max_channel_power

# Data settings
data_identifier = 'lidc'
preproc_folder = r'D:\dev_x\phiseg_log'
data_root = r'D:\dev_x\cv_uncertainty\qubiq\data_lidc.pickle'
dimensionality_mode = '2D'
image_size = (128, 128, 1)
nlabels = 2
num_labels_per_subject = 4

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': nlabels}

# training
optimizer = tf.train.AdamOptimizer
lr_schedule_dict = {0: 1e-3}
deep_supervision = True
batch_size = 12
num_iter = 200
annotator_range = [0]  # which annotators to actually use for training

# losses
KL_divergence_loss_weight = 1.0
exponential_weighting = True

residual_multinoulli_loss_weight = 1.0

# monitoring
do_image_summaries = True
rescale_RGB = False
validation_frequency = 20
validation_samples = 4
num_validation_images = 100 #'all'
tensorboard_update_frequency = 10

