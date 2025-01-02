# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
import warnings
import torch
import numpy as np

from datasets.base import DataLoader
import datasets.registry
from foundations import hparams
from foundations import paths
from foundations.step import Step
from models.base import Model
import models.registry
from platforms.platform import get_platform
from training.checkpointing import restore_checkpoint
from training import optimizers
from training import standard_callbacks
from training.metric_logger import MetricLogger


def introduce_label_noise_batch(labels: torch.Tensor, 
                                noise_level: float=0.2, 
                                num_classes: int=10) -> torch.Tensor:
    """Introduce label noise to a batch of labels using vectorized operations.

    Args:
        labels: The original labels.
        noise_level: The fraction of labels to be changed.
        num_classes: The number of classes in the dataset.

    Returns:
        The labels with noise introduced.
    """
    num_noisy_samples = int(noise_level * len(labels))
    if num_noisy_samples == 0:
        return labels

    # Select random indices for introducing noise
    indices = np.random.choice(len(labels), num_noisy_samples, replace=False)
    
    # Generate noisy labels
    origin_labels = labels[indices]
    noisy_labels = (origin_labels + torch.randint_like(origin_labels, low=1, high=num_classes)) % num_classes
    labels[indices] = noisy_labels

    return labels


def train(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None
):

    """The main training loop for this framework.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. The provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """

    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model.to(get_platform().torch_device)
    optimizer = optimizers.get_optimizer(training_hparams, model)
    step_optimizer = optimizer
    
    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed
    
    # get the number of classes
    n_classes = train_loader.dataset.num_classes()

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return

    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, logger)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)
            
            # Introduce label noise
            noisy_labels = introduce_label_noise_batch(
                labels, noise_level=training_hparams.noise_level, num_classes=n_classes
            )

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples), noisy_labels)
            loss.backward()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)

    get_platform().barrier()


def standard_train(
  model: Model,
  output_location: str,
  dataset_hparams: hparams.DatasetHparams,
  training_hparams: hparams.TrainingHparams,
  start_step: Step = None,
  verbose: bool = True,
  evaluate_every_epoch: bool = True
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (models.registry.exists(output_location, train_end_step) and
        get_platform().exists(paths.logger(output_location))): return

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    callbacks = standard_callbacks.standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch)
    train(training_hparams, model, train_loader, output_location, callbacks, start_step=start_step)
