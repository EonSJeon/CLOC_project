"""
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

RegressionModel class: a multi-layer perceptron (MLP) that supports:
    - Flexible hidden-layer configurations (including dropout and activation).
    - Poisson outputs (e.g. for count data).
    - Masked training for missing values.
    - Optional combination with an external “prior prediction” (either additive or multiplicative).
    - Keras-based training with early stopping or multiple initialization attempts.
    
Mathematical details are in RegressionModelDoc.md.
"""

import copy
import io
import logging
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .tools.abstract_classes import PredictorModel
from .tools.model_base_classes import ModelWithFitWithRetry, Reconstructable
from .tools.tf_losses import (
    masked_CC,
    masked_mse,
    masked_PoissonLL_loss,
    masked_R2,
)
from .tools.tf_tools import set_global_tf_eagerly_flag
from .tools.tools import getIsOk

logger = logging.getLogger(__name__)


class RegressionModel(tf.keras.layers.Layer, ModelWithFitWithRetry, Reconstructable):
    """
    A TensorFlow Keras-based multi-layer perceptron model (MLP) that implements
    flexible hidden layers, dropout, and specialized handling of:
      - Poisson distribution (with exponential activation),
      - Masked training for missing data,
      - Prior prediction injection (additive or multiplicative).

    In addition to the standard arguments (dimensions, layers, etc.), the model 
    includes features for training with multiple initializations (init_attempts)
    and basic early stopping (early_stopping_patience). It also supports an optional
    scheduler-based learning rate.

    Args:
        n_in (int):
            Input feature dimension.
        n_out (int):
            Output dimension (e.g., # of regression targets).
        units (list or tuple, optional):
            A list/tuple that describes the hidden-layer sizes (not including the final layer).
            Defaults to [] (no hidden layers).
        use_bias (bool or list, optional):
            Whether each Dense layer uses a bias term. If a list, it must match the number of layers.
            Defaults to False.
        dropout_rate (float or list, optional):
            The dropout probability for each hidden layer. Can be a scalar (applied to all hidden 
            layers) or a list per-layer. Defaults to 0 (no dropout).
        kernel_initializer (str or list, optional):
            Keras initializer(s). Defaults to 'glorot_uniform'.
        bias_initializer (str or list, optional):
            Keras initializer(s) for bias. Defaults to 'zeros'.
        kernel_regularizer_name (str or list, optional):
            Name of Keras regularizer (e.g., 'l1', 'l2', 'l1_l2'), or None to skip. Defaults to None.
        kernel_regularizer_args (dict or list, optional):
            Dict of arguments to pass to the regularizer constructor. Defaults to {}.
        bias_regularizer_name (str or list, optional):
            Same logic for bias regularizer. Defaults to None.
        bias_regularizer_args (dict or list, optional):
            Dict of arguments for the bias regularizer. Defaults to {}.
        activation (str or list, optional):
            The activation(s) for the hidden layers, e.g. "relu". Defaults to "linear".
        output_activation (str, optional):
            The activation used for the final layer. Defaults to "linear" except for out_dist='poisson', 
            which is automatically set to "exponential".
        out_dist (str, optional):
            If "poisson", configures the final layer for count data. Defaults to None (Gaussian).
        has_prior_pred (bool, optional):
            If True, the model expects an additional "prior_pred" input. Defaults to False.
        prior_pred_op (str, optional):
            The operation to combine prior_pred with the MLP output. "add" or "multiply". If None, 
            defaults to "multiply" for 'poisson' or "add" otherwise.
        name (str, optional):
            A prefix for the layer and sub-layers. Defaults to "reg_".
        log_dir (str, optional):
            Directory for saving training logs (e.g., TensorBoard). Defaults to "".
        optimizer_name (str or callable, optional):
            Name of a built-in Keras optimizer (e.g. "Adam"), or an optimizer constructor.
            Defaults to "Adam".
        optimizer_args (dict, optional):
            Arguments for the optimizer (learning_rate, etc.). Defaults to None.
        lr_scheduler_name (str or callable, optional):
            If set, uses a tf.keras.optimizers.schedules.* learning rate scheduler by that name,
            or a direct constructor. Defaults to None.
        lr_scheduler_args (dict, optional):
            Arguments for the LR scheduler. Defaults to None.
        missing_marker (float, optional):
            A special value indicating missing data in X_out or prior predictions. If not None, 
            a masked loss is used (only valid entries are included in the loss). Defaults to None.

    Attributes:
        model (tf.keras.Model): 
            The compiled Keras model (internal MLP plus optional prior input merge).
        layers (list):
            A list of the Dense + Dropout layers composing the MLP.
    """

    def __init__(
        self,
        n_in,
        n_out,
        units=[],
        use_bias=False,
        dropout_rate=0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer_name=None,
        kernel_regularizer_args={},
        bias_regularizer_name=None,
        bias_regularizer_args={},
        activation="linear",
        output_activation=None,
        out_dist=None,
        has_prior_pred=False,
        prior_pred_op=None,
        name="reg_",
        log_dir="",
        optimizer_name="Adam",
        optimizer_args=None,
        lr_scheduler_name=None,
        lr_scheduler_args=None,
        missing_marker=None,
    ):
        """
        Constructor for the RegressionModel class. See class docstring for full details.
        """
        self.constructor_kwargs = {
            "n_in": n_in,
            "n_out": n_out,
            "units": units,
            "dropout_rate": dropout_rate,
            "use_bias": use_bias,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
            "kernel_regularizer_name": kernel_regularizer_name,
            "kernel_regularizer_args": kernel_regularizer_args,
            "bias_regularizer_name": bias_regularizer_name,
            "bias_regularizer_args": bias_regularizer_args,
            "activation": activation,
            "output_activation": output_activation,
            "out_dist": out_dist,
            "has_prior_pred": has_prior_pred,
            "prior_pred_op": prior_pred_op,
            "name": name,
            "log_dir": log_dir,
            "optimizer_name": optimizer_name,
            "optimizer_args": optimizer_args,
            "lr_scheduler_name": lr_scheduler_name,
            "lr_scheduler_args": lr_scheduler_args,
            "missing_marker": missing_marker,
        }
        super(RegressionModel, self).__init__(name=name)

        def ensure_is_list(v):
            """Helper to ensure that 'v' is turned into a list if it isn't already."""
            v = copy.deepcopy(v)
            if "ListWrapper" in str(type(v)):
                v = list(v)
            if isinstance(v, tuple):
                v = list(v)
            if not isinstance(v, (list, tuple)):
                v = [v]
            return v

        units = ensure_is_list(units)
        # The final layer dimension
        units.append(n_out)

        if output_activation is None:
            # default: 'linear' for normal, 'exponential' for Poisson
            output_activation = "linear" if out_dist != "poisson" else "exponential"

        self.n_in = n_in
        self.n_out = n_out
        self.num_layers = len(units)
        self.layers = []
        self.units = units
        self.dropout_rate = ensure_is_list(dropout_rate)
        self.use_bias = ensure_is_list(use_bias)
        self.kernel_initializer = ensure_is_list(kernel_initializer)
        self.bias_initializer = ensure_is_list(bias_initializer)
        self.kernel_regularizer_name = ensure_is_list(kernel_regularizer_name)
        self.kernel_regularizer_args = ensure_is_list(kernel_regularizer_args)
        self.bias_regularizer_name = ensure_is_list(bias_regularizer_name)
        self.bias_regularizer_args = ensure_is_list(bias_regularizer_args)
        self.activation = ensure_is_list(activation)
        self.output_activation = output_activation
        self.out_dist = out_dist
        self.has_prior_pred = has_prior_pred

        # Decide how to fuse prior predictions
        if prior_pred_op is None:
            prior_pred_op = "multiply" if self.out_dist == "poisson" else "add"
        self.prior_pred_op = prior_pred_op

        self.name_prefix = name
        self.log_dir = log_dir
        self.logsub_dir = ""
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_args = lr_scheduler_args if lr_scheduler_args is not None else {}
        self.missing_marker = missing_marker

        self.build()

    def build(self):
        """
        Constructs the Dense (+ optional Dropout) layers 
        and compiles a tf.keras.Model internally.
        """
        def get_nth_or_last_elem(L, ci):
            """
            Utility to fetch either L[ci] or the last element if ci >= len(L).
            """
            return L[int(np.min((ci, len(L) - 1)))]

        # Primary input
        self.inputs = tf.keras.Input(
            shape=(self.n_in,), name=f"{self.name_prefix}input"
        )
        x = self.inputs

        for ci in range(self.num_layers):
            # Decide activation
            if ci == (self.num_layers - 1):
                thisActivation = self.output_activation
            elif len(self.activation) > ci:
                thisActivation = self.activation[ci]
            else:
                thisActivation = self.activation[-1]

            # Possibly build kernel/bias regularizers
            kernel_reg_name = get_nth_or_last_elem(self.kernel_regularizer_name, ci)
            kernel_reg_args = get_nth_or_last_elem(self.kernel_regularizer_args, ci)
            kernel_regularizer = None
            if kernel_reg_name is not None:
                kernel_regularizer = getattr(
                    tf.keras.regularizers, kernel_reg_name
                )(**kernel_reg_args)

            bias_reg_name = get_nth_or_last_elem(self.bias_regularizer_name, ci)
            bias_reg_args = get_nth_or_last_elem(self.bias_regularizer_args, ci)
            bias_regularizer = None
            if bias_reg_name is not None:
                bias_regularizer = getattr(
                    tf.keras.regularizers, bias_reg_name
                )(**bias_reg_args)

            nUnits = copy.copy(self.units[ci])

            # Create Dense
            layer = tf.keras.layers.Dense(
                nUnits,
                use_bias=get_nth_or_last_elem(self.use_bias, ci),
                kernel_initializer=get_nth_or_last_elem(self.kernel_initializer, ci),
                bias_initializer=get_nth_or_last_elem(self.bias_initializer, ci),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activation=thisActivation,
                name=f"{self.name_prefix}dense_{ci+1}",
            )
            self.layers.append(layer)
            x = layer(x)

            # Dropout if needed
            this_drop = get_nth_or_last_elem(self.dropout_rate, ci)
            if this_drop > 0 and (
                ci < (self.num_layers - 1) or len(self.dropout_rate) >= self.num_layers
            ):
                drop_layer = tf.keras.layers.Dropout(
                    this_drop, name=f"{self.name_prefix}dropout_{ci+1}"
                )
                self.layers.append(drop_layer)
                x = drop_layer(x)

        self.outputs = x

        # If prior is used, add another input
        if self.has_prior_pred:
            self.prior_pred = tf.keras.Input(
                shape=(self.n_out,), name=f"{self.name_prefix}prior_pred"
            )
            if self.prior_pred_op == "add":
                self.outputs = self.outputs + self.prior_pred
            elif self.prior_pred_op == "multiply":
                self.outputs = self.outputs * self.prior_pred
            else:
                raise ValueError("prior_pred_op must be 'add' or 'multiply'.")

            self.model = tf.keras.models.Model(
                inputs=[self.inputs, self.prior_pred],
                outputs=self.outputs
            )
        else:
            self.model = tf.keras.models.Model(
                inputs=self.inputs, outputs=self.outputs
            )
        self.compile()

    def compile(self):
        """
        Compiles the Keras model with masked or standard loss:
          - 'poisson' if self.out_dist == "poisson"
          - 'mse' otherwise
        Also includes masked variants if self.missing_marker is set,
        plus R^2 and CC as metrics for regression.
        """
        # Choose base loss
        if self.out_dist == "poisson":
            if self.missing_marker is None:
                loss = "poisson"
            else:
                loss = masked_PoissonLL_loss(self.missing_marker)
        else:
            if self.missing_marker is None:
                loss = "mse"
            else:
                loss = masked_mse(self.missing_marker)

        # Add metrics for regression
        metrics = []
        if self.out_dist != "poisson":
            metrics.append(masked_R2(self.missing_marker))
            metrics.append(masked_CC(self.missing_marker))
        metrics.append(loss)

        # Possibly build LR schedule
        if isinstance(self.lr_scheduler_name, str):
            if hasattr(tf.keras.optimizers.schedules, self.lr_scheduler_name):
                lr_scheduler_constructor = getattr(
                    tf.keras.optimizers.schedules, self.lr_scheduler_name
                )
            else:
                raise ValueError(
                    f"Learning rate scheduler {self.lr_scheduler_name} not recognized."
                )
        else:
            lr_scheduler_constructor = self.lr_scheduler_name

        if isinstance(self.optimizer_name, str):
            if self.optimizer_name.lower() == "adam":
                self.optimizer_name = "Adam"
            if hasattr(tf.keras.optimizers, self.optimizer_name):
                optimizer_constructor = getattr(tf.keras.optimizers, self.optimizer_name)
            else:
                raise ValueError(
                    f"Optimizer '{self.optimizer_name}' not recognized."
                )
        else:
            optimizer_constructor = self.optimizer_name

        # if we do have a schedule
        if lr_scheduler_constructor is not None:
            if (
                "learning_rate" in self.optimizer_args
                and "initial_learning_rate" not in self.lr_scheduler_args
            ):
                self.lr_scheduler_args["initial_learning_rate"] = self.optimizer_args["learning_rate"]
            lr_sched = lr_scheduler_constructor(**self.lr_scheduler_args)
            self.optimizer_args["learning_rate"] = lr_sched

        optimizer = optimizer_constructor(**self.optimizer_args)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.run_eagerly = False

    def get_config(self):
        """
        Returns serialization config. For Keras model saving/loading.
        """
        config = super(RegressionModel, self).get_config()
        param_names = [
            "n_in",
            "n_out",
            "units",
            "kernel_initializer",
            "kernel_regularizer_name",
            "kernel_regularizer_args",
            "use_bias",
            "bias_regularizer_name",
            "bias_regularizer_args",
            "activation",
            "output_activation",
            "out_dist",
            "has_prior_pred",
            "prior_pred_op",
            "log_dir",
            "missing_marker",
        ]
        for nm in param_names:
            config[nm] = getattr(self, nm)
        config.update({"name": self.name_prefix})
        return config

    def apply_func(self, inputs, name_scope=None):
        """
        For advanced usage: manually run the forward pass on 'inputs'.
        If has_prior_pred=True, 'inputs' must be [x_in, prior].
        """
        with tf.name_scope(self.name_prefix if name_scope is None else name_scope):
            if self.has_prior_pred:
                x_in, prior_pred = inputs[0], inputs[1]
            else:
                x_in = inputs

            out = x_in
            for ci, layer in enumerate(self.layers):
                out = layer(out)
            if self.has_prior_pred:
                if self.prior_pred_op == "add":
                    out += prior_pred
                elif self.prior_pred_op == "multiply":
                    out *= prior_pred
        return out

    def setTrainable(self, trainable):
        """
        Toggles trainability for all layers, then re-compiles.
        """
        self.trainable = trainable
        self.compile()

    def fit(
        self,
        X_in,
        X_out,
        prior_pred=None,
        X_in_val=None,
        X_out_val=None,
        prior_pred_val=None,
        epochs=100,
        batch_size=None,
        verbose=False,
        init_attempts=1,
        max_attempts=1,
        early_stopping_patience=3,
        start_from_epoch=0,
        early_stopping_measure="loss",
    ):
        """
        Training with optional early stopping, multiple init attempts, etc.
        X_in, X_out shape: (dim, n_samples)
        """
        def prep_IO(X_in, X_out, prior_pred, label="training"):
            # Build mask
            if self.missing_marker is not None:
                isOk = np.logical_not(np.any(X_out == self.missing_marker, axis=0))
                if not np.all(isOk):
                    logger.info(
                        f"{np.sum(isOk)}/{len(isOk)} samples (~{100*np.sum(isOk)/len(isOk):.1f}%) "
                        f"available for {label} (others missing, marker={self.missing_marker})."
                    )
            else:
                isOk = np.ones(X_out.shape[1], dtype=bool)

            X_in_sel  = X_in[:, isOk].T
            X_out_sel = X_out[:, isOk].T

            if self.has_prior_pred:
                if prior_pred is None:
                    # default prior => zero for regression or 1 for poisson if you prefer
                    if self.out_dist == "poisson":
                        prior_pred = np.ones_like(X_out)
                    else:
                        prior_pred = np.zeros_like(X_out)
                prior_pred_sel = prior_pred[:, isOk].T
                return ([X_in_sel, prior_pred_sel], X_out_sel)
            else:
                return (X_in_sel, X_out_sel)

        # Prep data
        train_in, train_out = prep_IO(X_in, X_out, prior_pred, "training")
        if X_in_val is not None:
            val_in, val_out = prep_IO(X_in_val, X_out_val, prior_pred_val, "validation")
            validation_data = (val_in, val_out)
        else:
            validation_data = None

        # Try multiple training attempts if needed
        fitOk, attempt = False, 0
        history = None
        while not fitOk and attempt < max_attempts:
            attempt += 1
            history = self.fit_with_retry(
                init_attempts=init_attempts,
                early_stopping_patience=early_stopping_patience,
                early_stopping_measure=early_stopping_measure,
                start_from_epoch=start_from_epoch,
                x=train_in,
                y=train_out,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                verbose=verbose,
            )
            # check for NaN
            fitOk = not np.isnan(history.history["loss"][-1])
            if not fitOk and attempt < max_attempts:
                logger.info(f"NaN loss found. Retrying training attempt {attempt+1}")

        # Summarize
        batch_count = history.params["steps"]
        logger.info(
            f"Finished training. We had {batch_count} steps, batch_size ~ {int(train_out.shape[0]/batch_count)}. "
            f"(n_in={X_in.shape[0]}, n_out={X_out.shape[0]})"
        )
        return history

    def predict(self, X_in, prior_pred=None):
        """
        Forward pass => returns (n_out, n_samples).
        For Poisson, returns the predicted mean rate. 
        If has_prior_pred, user can pass prior_pred with shape (n_out, n_samples).
        """
        eagerly_flag_backup = set_global_tf_eagerly_flag(False)
        if self.has_prior_pred and prior_pred is None:
            # if no prior => default
            if self.out_dist == "poisson":
                prior_pred = np.ones((self.n_out, X_in.shape[1]))
            else:
                prior_pred = np.zeros((self.n_out, X_in.shape[1]))

        # Keras fix
        if not hasattr(self.model, "_predict_counter"):
            from tensorflow.keras.backend import variable
            self.model._predict_counter = variable(0)

        if self.has_prior_pred:
            # shape => [X_in, prior_pred]
            preds = self.model.predict([X_in.T, prior_pred.T])
            preds = preds.T
        else:
            preds = self.model.predict(X_in.T).T

        set_global_tf_eagerly_flag(eagerly_flag_backup)
        return preds

    def plot_comp_graph(
        self,
        savepath="model_graph",
        saveExtensions=None,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True,
    ):
        """
        Visualize with tf.keras.utils.plot_model. By default saves to model_graph.png
        """
        if saveExtensions is None:
            saveExtensions = ["png"]
        saveExtensions = [ex for ex in saveExtensions if ex != "svg"]
        for fmt in saveExtensions:
            try:
                tf.keras.utils.plot_model(
                    self.model,
                    to_file=f"{savepath}.{fmt}",
                    show_shapes=show_shapes,
                    show_layer_names=show_layer_names,
                    expand_nested=expand_nested,
                )
                logger.info(f"Saved model graph as {savepath}.{fmt}")
            except Exception as e:
                logger.error(e)




# class DRModel(PredictorModel):
#     """A class that implements non-linear direction regression model based on RegressionModel"""

#     def __init__(self, log_dir=""):  # If not empty, will store tensorboard logs
#         self.log_dir = log_dir

#     @staticmethod
#     def parse_method_code(
#         methodCode, YType=None, ZType=None, Z=None, missing_marker=None
#     ):
#         Dz_args = {}
#         Dy_args = {}
#         if "HL" in methodCode:
#             regex = r"([Dz|Dy|]*)(\d+)HL(\d+)U"  # 1HL100U
#             matches = re.finditer(regex, methodCode)
#             for matchNum, match in enumerate(matches, start=1):
#                 var_names, hidden_layers, hidden_units = match.groups()
#             hidden_layers = int(hidden_layers)
#             hidden_units = int(hidden_units)
#             activation = "relu"
#             NL_args = {
#                 "use_bias": True,
#                 "units": [hidden_units] * hidden_layers,
#                 "activation": activation,
#             }
#             if var_names == "" or "Dz" in var_names:
#                 Dz_args = copy.deepcopy(NL_args)
#             if var_names == "" or "Dy" in var_names:
#                 Dy_args = copy.deepcopy(NL_args)
#         if "RGL" in methodCode:  # Regularize
#             regex = r"([Dz|Dy|]*)RGL(\d+)"  #
#             matches = re.finditer(regex, methodCode)
#             for matchNum, match in enumerate(matches, start=1):
#                 var_names, norm_num = match.groups()
#             if norm_num in ["1", "2"]:
#                 regularizer_name = "l{}".format(norm_num)
#             else:
#                 raise (Exception("Unsupported method code: {}".format(methodCode)))
#             lambdaVal = 0.01  # Default: 'l': 0.01
#             regex = r"L(\d+)e([-+])?(\d+)"  # 1e-2
#             matches = re.finditer(regex, methodCode)
#             for matchNum, match in enumerate(matches, start=1):
#                 m, sgn, power = match.groups()
#                 if sgn is not None and sgn == "-":
#                     power = -float(power)
#                 lambdaVal = float(m) * 10 ** float(power)
#             regularizer_args = {"l": lambdaVal}  # Default: 'l': 0.01
#             RGL_args = {
#                 "kernel_regularizer_name": regularizer_name,
#                 "kernel_regularizer_args": regularizer_args,
#                 "bias_regularizer_name": regularizer_name,
#                 "bias_regularizer_args": regularizer_args,
#             }
#             if var_names == "" or "Dz" in var_names:
#                 Dz_args.update(copy.deepcopy(RGL_args))
#             if var_names == "" or "Dy" in var_names:
#                 Dy_args.update(copy.deepcopy(RGL_args))

#         if ZType == "count_process":
#             Dz_args["use_bias"] = True
#             Dz_args["out_dist"] = "poisson"
#             Dz_args["output_activation"] = "exponential"
#         elif ZType == "cat":
#             isOkZ = getIsOk(Z, missing_marker)
#             ZClasses = np.unique(Z[:, np.all(isOkZ, axis=0)])
#             Dz_args["num_classes"] = len(ZClasses)
#             Dz_args["use_bias"] = True

#         return Dy_args, Dz_args

#     def fit(
#         self,
#         Y,
#         Z=None,
#         U=None,
#         batch_size=32,  # Each batch consists of this many blocks with block_samples time steps
#         epochs=250,  # Max number of epochs to go over the whole training data
#         Y_validation=None,  # if provided will use to compute loss on validation
#         Z_validation=None,  # if provided will use to compute loss on validation
#         U_validation=None,  # if provided will use to compute loss on validation
#         true_model=None,
#         missing_marker=None,  # Values of z that are equal to this will not be used
#         allowNonzeroCz2=True,
#         model2_Cz_Full=True,
#         skip_Cy=False,  # If true and only stage 1 (n1 >= nx), will not learn Cy (model will not have neural self-prediction ability)
#         clear_graph=True,  # If true will wipe the tf session before starting, so that variables names don't get numbers at the end and mem is preserved
#         YType=None,
#         ZType=None,
#         Dy_args={},
#         Dz_args={},
#     ):
#         if clear_graph:
#             tf.keras.backend.clear_session()

#         isOkY = getIsOk(Y, missing_marker)
#         isOkZ = getIsOk(Z, missing_marker)
#         isOkU = getIsOk(U, missing_marker)

#         if YType is None:  # Auto detect signal types
#             YType = autoDetectSignalType(Y)

#         if ZType is None:  # Auto detect signal types
#             ZType = autoDetectSignalType(Z)

#         if ZType == "cat":
#             ZClasses = np.unique(Z[:, np.all(isOkZ, axis=0)])

#         if YType == "count_process":
#             yDist = "poisson"
#         else:
#             yDist = None

#         if U is None:
#             U = np.empty(0)
#         nu = U.shape[0]

#         ny, Ndat = Y.shape[0], Y.shape[1]
#         if Z is not None:
#             nz, NdatZ = Z.shape[0], Z.shape[1]

#         if nu > 0:
#             YU = np.concatenate([Y, U], axis=0)
#             if Y_validation is not None:
#                 YU_validation = np.concatenate([Y_validation, U_validation], axis=0)
#             else:
#                 YU_validation = None
#         else:
#             YU = Y
#             YU_validation = Y_validation

#         logger.info("Learning regression")
#         this_log_dir = "" if self.log_dir == "" else os.path.join(self.log_dir, "Dz")
#         reg_args = copy.deepcopy(Dz_args)
#         model_Dz = RegressionModel(
#             ny + nu, nz, log_dir=this_log_dir, missing_marker=missing_marker, **reg_args
#         )
#         history_Dz = model_Dz.fit(
#             YU, Z, X_in_val=YU_validation, X_out_val=Z_validation, epochs=epochs
#         )
#         self.logs = {"model_Dz": history_Dz}

#         self.Dz_args = Dz_args
#         self.Dy_args = Dy_args

#         self.ny = ny
#         self.nz = nz
#         self.nu = nu

#         self.model_Dz = model_Dz

#         self.missing_marker = missing_marker
#         self.batch_size = batch_size

#     def discardModels(self):
#         if self.nz > 0:
#             self.model_Dz = self.model_Dz.model.get_weights()

#     def restoreModels(self):
#         if self.nz > 0:
#             w = self.model_Dz
#             reg_args = copy.deepcopy(self.Dz_args)
#             self.model_Dz = RegressionModel(
#                 self.ny + self.nu,
#                 self.nz,
#                 missing_marker=self.missing_marker,
#                 **reg_args,
#             )
#             self.model_Dz.model.set_weights(w)

#     def predict(self, Y, U=None):
#         """
#         Y: sample x ny
#         U: sample x nu
#         """

#         eagerly_flag_backup = set_global_tf_eagerly_flag(False)
#         Ndat = Y.shape[0]

#         if U is None and self.nu > 0:
#             U = np.zeros((Ndat, self.nu))
#         if self.nu > 0:
#             YU = np.concatenate([Y, U], axis=1)
#         else:
#             YU = Y

#         allZp = self.model_Dz.predict(YU.T)
#         allYp = None
#         allXp = None

#         if self.nz > 0 and allZp is not None:
#             if len(allZp.shape) == 2:
#                 allZp = allZp.T
#             else:
#                 allZp = allZp.transpose([1, 0, 2])

#         set_global_tf_eagerly_flag(eagerly_flag_backup)
#         return allZp, allYp, allXp
