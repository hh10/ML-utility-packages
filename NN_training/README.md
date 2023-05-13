# NN training boilerplate

## Purpose
- Separate logic from the standard structure and utils required for most trainings,
- Reduce code duplication in NN trainings,
- Easier logging of training config and results.

## Description
The [trainer.py](trainer.py) module implements `Trainer` and `TrainingRunner` classes.
- Trainer class:
    - stores the training config, hyperparameters, etc. required to define and reproduce a training. These are stored as a dataclass and logged in a file before a training run.
    - it has an abstract method called `loss_and_accuracy` which should be implemented by a child class. It receives data from dataloader as its argument and should return loss and accuracy. The former is then used by the member optimizer for training.  
    - this method can also update various losses and accuracies as achieved during training and testing. A `StatsCalc` (which can be extended to include more stats per entity) is instantied and its checkpoint is stored per epoch start and end for each loss and accuracy version logged in `loss_and_accuracy` method. The TrainingRunner then plots these checkpoints as progress at every epoch end.
- TrainingRunner class:
    - takes in a list of Trainer instances (say, for multi-head/objective/loss trainings) that are executed in the same order as provided.
    - the training config as specified is logged in a file for reference.

## Usage
An example of using Trainer and TrainingRunner for classification is implemented in [classification_example.py](classification_example.py). 
