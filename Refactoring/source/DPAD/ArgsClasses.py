# file: ArgsClasses.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from enum import Enum

__all__ = [
    "DistributionType",
    "LayerConfig",
    "DPADConfig",
    "Activation",
    "KernelInitializer",
    "SignalType",
    "OptimizerName",
    "OptimizerConfig",
]

class DistributionType(Enum):
    POISSON = "poisson"
    NORMAL = "normal"
    GAUSSIAN = "gaussian"

class Activation(Enum):
    RELU = "relu"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"  

class KernelInitializer(Enum):
    """
    A kernel initializer is a method used to set the initial values of the weights 
    (kernels) of a neural network layer before training begins. The choice of initializer 
    affects how well and how quickly the network learns, as it influences the 
    distribution of activation values and gradients across layers.

    Key considerations for kernel initialization:
    - Preventing vanishing or exploding gradients: Poor initialization can lead to 
    gradients becoming too small (vanishing) or too large (exploding), making 
    training difficult.
    - Maintaining signal propagation: The initializer should help maintain a reasonable 
    flow of information through the network.
    - Adapting to activation functions: Different activation functions work best with 
    different initializations.

    The following kernel initializers are commonly used:

    1. ORTHOGONAL ("orthogonal"):
    - Initializes the weight matrix using an orthogonal matrix (a matrix whose rows and 
        columns are orthonormal).
    - Helps maintain the norm of activations across layers, particularly useful in deep networks 
        and recurrent neural networks (RNNs).
    - Prevents vanishing/exploding gradients by preserving variance during forward and 
        backward propagation.
    - Often used in RNNs and deep architectures.

    2. GLOROT_UNIFORM ("glorot_uniform"):
    - Also known as Xavier initialization.
    - Draws values from a uniform distribution with limits determined by the number 
        of input and output neurons: 
        range = [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
    - Ensures that the variance of activations remains roughly the same across layers, 
        helping to stabilize training.
    - Commonly used in feedforward and convolutional neural networks (CNNs).

    3. ZEROS ("zeros"):
    - Initializes all weights to zero.
    - Generally not recommended for most layers, as it prevents neurons from 
        learning independently (they all receive the same gradients).
    - Sometimes used for bias initialization or in specific constrained learning scenarios.
    """
    ORTHOGONAL = "orthogonal"
    GLOROT_UNIFORM = "glorot_uniform"
    ZEROS = "zeros"

class SignalType(Enum):
    CONTINUOUS = "cont"
    CATEGORICAL = "cat"
    COUNT_PROCESS = "count_process"

class OptimizerName(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSProp"
    SGD = "SGD"

@dataclass
class OptimizerConfig:
    name: OptimizerName = OptimizerName.ADAM
    args: Optional[Dict] = None
    lr_scheduler_name: Optional[str] = None  # Could be tokenized as an enum too
    lr_scheduler_args: Optional[Dict] = None

@dataclass
class LayerConfig:
    use_bias: bool = True
    units: List[int] = field(default_factory=list)
    activation: Activation = Activation.RELU
    output_activation: Activation = Activation.LINEAR
    kernel_initializer: Optional[KernelInitializer] = None
    unifiedAK: bool = False
    dummy: bool = False
    dropout_rate: Optional[float] = None
    kernel_regularizer_name: Optional[str] = None
    kernel_regularizer_args: Optional[Dict] = None
    bias_regularizer_name: Optional[str] = None
    bias_regularizer_args: Optional[Dict] = None
    out_dist: Optional[str] = None

@dataclass
class DPADConfig:
    # Per-layer configurations.
    A1: LayerConfig = field(default_factory=LayerConfig)
    K1: LayerConfig = field(default_factory=LayerConfig)
    Cy1: LayerConfig = field(default_factory=LayerConfig)
    Cz1: LayerConfig = field(default_factory=LayerConfig)
    A2: LayerConfig = field(default_factory=LayerConfig)
    K2: LayerConfig = field(default_factory=LayerConfig)
    Cy2: LayerConfig = field(default_factory=LayerConfig)
    Cz2: LayerConfig = field(default_factory=LayerConfig)
    
    # Other settings.
    init_method: Optional[str] = None
    init_attempts: int = 1
    batch_size: Optional[int] = None
    early_stopping_patience: int = 3
    early_stopping_measure: str = "loss"
    start_from_epoch_rnn: int = 0
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    steps_ahead: List[int] = field(default_factory=lambda: [1])
    steps_ahead_loss_weights: Optional[List[float]] = None
    
    # Architecture flags.
    model1_Cy_Full: bool = False
    model2_Cz_Full: bool = False
    linear_cell: bool = False
    LSTM_cell: bool = False
    bidirectional: bool = False
    allow_nonzero_Cz2: bool = True
    has_Dyz: bool = False
    skip_Cy: bool = False
    zscore_inputs: bool = True

    # Optional references to your data types
    YType: Optional[SignalType] = None
    ZType: Optional[SignalType] = None
    UType: Optional[SignalType] = None

    # Additional fields for DPAD fine-tuning
    nx: Optional[int] = None
    n1: Optional[int] = None
    max_attempts: int = 10
    regression_init_method: Optional[str] = None
    init_model: Optional[object] = None
    linear_cell_for_test_only: bool = False  # Example usage if needed
    enable_forward_pred: bool = False
    throw_on_fail: bool = False
    clear_graph: bool = False
    use_existing_prep_models: bool = True

@dataclass
class FitConfig(DPADConfig):
    """An example 'FitConfig' that inherits from DPADConfig,
    adding or refining fields that are specific to fitting."""
    
    # These fields might override or extend what DPADConfig provides.
    epochs: int = 2500
    max_attempts: int = 10
    create_val_from_training: bool = False
    validation_set_ratio: float = 0.2