import numpy as np

import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from model_architectures import ConvolutionalNetwork

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed



if args.dataset_name == 'emnist':
    train_data = data_providers.EMNISTDataProvider('train', batch_size=args.batch_size,
                                                   rng=rng,
                                                   flatten=False)  # initialize our rngs using the argument set seed
    val_data = data_providers.EMNISTDataProvider('valid', batch_size=args.batch_size,
                                                 rng=rng,
                                                 flatten=False)  # initialize our rngs using the argument set seed
    test_data = data_providers.EMNISTDataProvider('test', batch_size=args.batch_size,
                                                  rng=rng,
                                                  flatten=False)  # initialize our rngs using the argument set seed
    num_output_classes = train_data.num_classes



#custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
#    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
#    dim_reduction_type=args.dim_reduction_type, num_filters=args.num_filters, num_layers=args.num_layers,
#    use_bias=False,
#    num_output_classes=num_output_classes)

#conv_experiment = ExperimentBuilder(network_model=custom_conv_net, use_gpu=args.use_gpu,
#                                    experiment_name=args.experiment_name,
#                                    num_epochs=args.num_epochs,
#                                    weight_decay_coefficient=args.weight_decay_coefficient,
#                                    continue_from_epoch=args.continue_from_epoch,
#                                    train_data=train_data, val_data=val_data,
#                                    test_data=test_data)  # build an experiment object
#experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics

diego_conv_net = DiegoConvolutionalNetwork(
            input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
            num_output_classes=num_output_classes,
            num_filters=args.num_filters,
            use_bias=False)

diego_conv_experiment = ExperimentBuilder(network_model=diego_conv_net, use_gpu=args.use_gpu,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data, reg=args.reg)  # build an experiment object

experiment_metrics, test_metrics = diego_conv_experiment.run_experiment()  # run experiment and return experiment metrics
