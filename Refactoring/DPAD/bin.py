
# DPADModel.py
def getLossLogStr(trueVals, predVals, steps, sigType, lossFuncs):
    if not isinstance(trueVals, (list, tuple)):
        trueVals = [trueVals]
    if not isinstance(predVals, (list, tuple)):
        predVals = [predVals]
    if steps is None:
        steps = [1]
    strs = []
    for ind in range(len(predVals)):
        trueVal = trueVals[ind % len(trueVals)]
        predVal = predVals[ind % len(predVals)]
        step = steps[ind % len(steps)]
        lossVals = [
            (
                lossFunc(trueVal, predVal.T)
                if sigType in ["cont", "count_process"]
                else lossFunc(trueVal, predVal.transpose([1, 0, 2]))
            )
            for lossFunc in lossFuncs
        ]
        strs.append(
            f"{step}-step: "
            + ", ".join(
                [
                    f"{lossFunc.__name__}={lossVal:.3g}"
                    for lossFunc, lossVal in zip(lossFuncs, lossVals)
                ]
            )
        )
    return "\n".join(strs)

## in DPAD class
@staticmethod
    def prepare_args(methodCode):
        """Parses a method code and preprares the arguments for the DPADModel constructor.

        Args:
            methodCode (string): DPAD method code string. For example can be "DPAD_uAKCzCy2HL128U"
                for a fully nonlinear model with 2 hidden layers of 128 units for each model parameter.

        Returns:
            kwargs: dict of arguments for the DPADModel constructor.
        """
        A1_args = {}
        K1_args = {}
        Cy1_args = {}
        Cz1_args = {}
        A2_args = {}
        K2_args = {}
        Cy2_args = {}
        Cz2_args = {}
        if "HL" in methodCode or "NonLin" in methodCode:
            regex = (
                r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)(\d+)HL(\d+)U"  # Cz1HL64U
            )
            if len(re.findall(regex, methodCode)) > 0:
                matches = re.finditer(regex, methodCode)
                for matchNum, match in enumerate(matches, start=1):
                    var_names, hidden_layers, hidden_units = match.groups()
                hidden_layers = int(hidden_layers)
                hidden_units = int(hidden_units)
            else:
                regex = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)NonLin"  # CzNonLin
                matches = re.finditer(regex, methodCode)
                for matchNum, match in enumerate(matches, start=1):
                    var_names = match.groups()
                hidden_layers = 3  # Default
                hidden_units = 64  # Default
            activation = "relu"
            NL_args = {
                "use_bias": True,
                "units": [hidden_units] * hidden_layers,
                "activation": activation,
            }
            if (
                var_names == ""
                or "A1" in var_names
                or ("A" in var_names and "A2" not in var_names)
            ):
                A1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "A2" in var_names
                or ("A" in var_names and "A1" not in var_names)
            ):
                A2_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "K1" in var_names
                or ("K" in var_names and "K2" not in var_names)
            ):
                K1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "K2" in var_names
                or ("K" in var_names and "K1" not in var_names)
            ):
                K2_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cy1" in var_names
                or ("Cy" in var_names and "Cy2" not in var_names)
            ):
                Cy1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cy2" in var_names
                or ("Cy" in var_names and "Cy1" not in var_names)
            ):
                Cy2_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cz1" in var_names
                or ("Cz" in var_names and "Cz2" not in var_names)
            ):
                Cz1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cz2" in var_names
                or ("Cz" in var_names and "Cz1" not in var_names)
            ):
                Cz2_args = copy.copy(NL_args)
        if (
            "OGInit" in methodCode
        ):  # Use initializers similar to the default keras LSTM initializers
            A1_args["kernel_initializer"] = "orthogonal"
            K1_args["kernel_initializer"] = "glorot_uniform"
            Cy1_args["kernel_initializer"] = "glorot_uniform"
            Cy1_args["kernel_initializer"] = "glorot_uniform"
            A2_args["kernel_initializer"] = "orthogonal"
            K2_args["kernel_initializer"] = "glorot_uniform"
            Cy2_args["kernel_initializer"] = "glorot_uniform"
            Cy2_args["kernel_initializer"] = "glorot_uniform"
        if "AKerIn0" in methodCode:  # Initialize A with zeros
            A1_args["kernel_initializer"] = "zeros"
            A2_args["kernel_initializer"] = "zeros"
        if "uAK" in methodCode:  # Unify A and K
            K1_args["unifiedAK"] = True
            K2_args["unifiedAK"] = True
        if "RGL" in methodCode:  # Regularize
            regex = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)RGLB?(\d+)?(Drop)?"  # _ARGL2_L1e5
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                var_names, norm_num, dropout = match.groups()
            # Param value
            lambdaVal = 0.01  # Default: 'l': 0.01
            regex = r"L(\d+)e([-+])?(\d+)"  # L1e-2
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                m, sgn, power = match.groups()
                if sgn is not None and sgn == "-":
                    power = -float(power)
                lambdaVal = float(m) * 10 ** float(power)
            RGL_args = {}
            if dropout is not None and dropout != "":  # Add dropout regularization
                RGL_args.update({"dropout_rate": lambdaVal})
            if norm_num is not None and norm_num != "":  # Add L1 or L2 regularization
                if norm_num in ["1", "2"]:
                    regularizer_name = "l{}".format(norm_num)
                else:
                    raise (Exception("Unsupported method code: {}".format(methodCode)))
                regularizer_args = {"l": lambdaVal}  # Default: 'l': 0.01
                RGL_args.update(
                    {
                        "kernel_regularizer_name": regularizer_name,
                        "kernel_regularizer_args": regularizer_args,
                    }
                )
                if "RGLB" in methodCode:  # Also regularize biases
                    # Reference for why usually biases don't need regularization:
                    # http://neuralnetworksanddeeplearning.com/chap3.html
                    RGL_args.update(
                        {
                            "bias_regularizer_name": regularizer_name,
                            "bias_regularizer_args": regularizer_args,
                        }
                    )
            if (
                var_names == ""
                or "A1" in var_names
                or ("A" in var_names and "A2" not in var_names)
            ):
                A1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "A2" in var_names
                or ("A" in var_names and "A1" not in var_names)
            ):
                A2_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "K1" in var_names
                or ("K" in var_names and "K2" not in var_names)
            ):
                K1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "K2" in var_names
                or ("K" in var_names and "K1" not in var_names)
            ):
                K2_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cy1" in var_names
                or ("Cy" in var_names and "Cy2" not in var_names)
            ):
                Cy1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cy2" in var_names
                or ("Cy" in var_names and "Cy1" not in var_names)
            ):
                Cy2_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cz1" in var_names
                or ("Cz" in var_names and "Cz2" not in var_names)
            ):
                Cz1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cz2" in var_names
                or ("Cz" in var_names and "Cz1" not in var_names)
            ):
                Cz2_args.update(copy.deepcopy(RGL_args))

        if "dummyA" in methodCode:  # Dummy A
            A1_args["dummy"] = True
            A2_args["dummy"] = True

        init_method = None
        if "init" in methodCode:  # Initialize
            regex = r"init(.+)"  #
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                initMethod = match.groups()
            if len(initMethod) > 0:
                init_method = initMethod[0]

        if "RTR" in methodCode:  # Retry initialization
            regex = r"RTR(\d+)"  # RTR2
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                init_attempts = match.groups()
            init_attempts = int(init_attempts[0])
        else:
            init_attempts = 1

        if "ErS" in methodCode:  # Early stopping
            regex = r"ErSV?(\d+)"  # ErS64
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                early_stopping_patience = match.groups()
                ErS_str = methodCode[match.span()[0] : match.span()[1]]
            early_stopping_patience = int(early_stopping_patience[0])
        else:
            early_stopping_patience = 3
            ErS_str = ""
        early_stopping_measure = "val_loss" if "ErSV" in ErS_str else "loss"

        if "MinEp" in methodCode:  # Minimum epochs
            regex = r"MinEp(\d+)"  # MinEp150
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                start_from_epoch_rnn = match.groups()
            start_from_epoch_rnn = int(start_from_epoch_rnn[0])
        else:
            start_from_epoch_rnn = 0

        if "BtcS" in methodCode:  # The batch_size
            regex = r"BtcS(\d+)"  # BtcS1
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                batch_size = match.groups()
            batch_size = int(batch_size[0])
        else:
            batch_size = None

        lr_scheduler_name = None
        lr_scheduler_args = None

        optimizer_name = "Adam"  # default
        optimizer_args = None
        optimizer_infos, matches = parseMethodCodeArgOptimizer(methodCode)
        if len(optimizer_infos) > 0:
            optimizer_info = optimizer_infos[0]
            if "optimizer_name" in optimizer_info:
                optimizer_name = optimizer_info["optimizer_name"]
            if "optimizer_args" in optimizer_info:
                optimizer_args = optimizer_info["optimizer_args"]
            if "scheduler_name" in optimizer_info:
                lr_scheduler_name = optimizer_info["scheduler_name"]
            if "scheduler_args" in optimizer_info:
                lr_scheduler_args = optimizer_info["scheduler_args"]

        steps_ahead, steps_ahead_loss_weights, matches = parseMethodCodeArgStepsAhead(
            methodCode
        )

        model1_Cy_Full = (
            "FCy" in methodCode or "FCyCz" in methodCode or "FCzCy" in methodCode
        )
        model2_Cz_Full = (
            "FCz" in methodCode or "FCyCz" in methodCode or "FCzCy" in methodCode
        )
        linear_cell = "LinCell" in methodCode
        LSTM_cell = "LSTM" in methodCode
        bidirectional = "xSmth" in methodCode or "bidir" in methodCode
        allow_nonzero_Cz2 = "Cz20" not in methodCode
        has_Dyz = "Dyz" in methodCode
        skip_Cy = "skipCy" in methodCode
        zscore_inputs = "nzs" not in methodCode

        kwargs = {
            "A1_args": A1_args,
            "K1_args": K1_args,
            "Cy1_args": Cy1_args,
            "Cz1_args": Cz1_args,
            "A2_args": A2_args,
            "K2_args": K2_args,
            "Cy2_args": Cy2_args,
            "Cz2_args": Cz2_args,
            "init_method": init_method,
            "init_attempts": init_attempts,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_measure": early_stopping_measure,
            "start_from_epoch_rnn": start_from_epoch_rnn,
            "model1_Cy_Full": model1_Cy_Full,
            "model2_Cz_Full": model2_Cz_Full,
            "linear_cell": linear_cell,
            "LSTM_cell": LSTM_cell,
            "bidirectional": bidirectional,
            "allow_nonzero_Cz2": allow_nonzero_Cz2,
            "has_Dyz": has_Dyz,
            "skip_Cy": skip_Cy,
            "steps_ahead": steps_ahead,  # List of ints (None take as [1]), indicating the number of steps ahead to generate from the model (used to construct training loss and predictions
            "steps_ahead_loss_weights": steps_ahead_loss_weights,  # Weight of each step ahead prediction in loss. If None, will give all steps ahead equal weight of 1.
            "zscore_inputs": zscore_inputs,
            "optimizer_name": optimizer_name,
            "optimizer_args": optimizer_args,
            "lr_scheduler_name": lr_scheduler_name,
            "lr_scheduler_args": lr_scheduler_args,
        }
        if A1_args == A2_args:
            kwargs["A_args"] = A1_args
            del kwargs["A1_args"], kwargs["A2_args"]
        if K1_args == K2_args:
            kwargs["K_args"] = K1_args
            del kwargs["K1_args"], kwargs["K2_args"]
        if Cy1_args == Cy2_args:
            kwargs["Cy_args"] = Cy1_args
            del kwargs["Cy1_args"], kwargs["Cy2_args"]
        if Cz1_args == Cz2_args:
            kwargs["Cz_args"] = Cz1_args
            del kwargs["Cz1_args"], kwargs["Cz2_args"]
        return kwargs

def add_default_param_args(
        self,
        A1_args={},
        K1_args={},
        Cy1_args={},
        Cz1_args={},
        A2_args={},
        K2_args={},
        Cy2_args={},
        Cz2_args={},
        yDist=None,
        zDist=None,
    ):
        LinArgs = {
            "units": [],
            "use_bias": False,
            "activation": "linear",
            "output_activation": "linear",
        }
        for f, v in LinArgs.items():
            if f not in A1_args:
                A1_args[f] = v
            if f not in K1_args:
                K1_args[f] = v
            if f not in Cy1_args:
                Cy1_args[f] = v
            if f not in Cz1_args:
                Cz1_args[f] = v
            if f not in A2_args:
                A2_args[f] = v
            if f not in K2_args:
                K2_args[f] = v
            if f not in Cy2_args:
                Cy2_args[f] = v
            if f not in Cz2_args:
                Cz2_args[f] = v

        if yDist == "poisson":
            Cy1_args["out_dist"] = "poisson"
            Cy1_args["output_activation"] = "exponential"
            Cy2_args["out_dist"] = "poisson"
            Cy2_args["output_activation"] = "exponential"

        if zDist == "poisson":
            Cz1_args["out_dist"] = "poisson"
            Cz1_args["output_activation"] = "exponential"
            Cz2_args["out_dist"] = "poisson"
            Cz2_args["output_activation"] = "exponential"

        if "unifiedAK" not in K1_args:
            K1_args["unifiedAK"] = False
        if "unifiedAK" not in K2_args:
            K2_args["unifiedAK"] = False
        return (
            A1_args,
            K1_args,
            Cy1_args,
            Cz1_args,
            A2_args,
            K2_args,
            Cy2_args,
            Cz2_args,
        )

def prep_observation_for_training(self, Y, YType):
        """Prepares the output distribution depending of signal type, a version of the output loss function,
        and appropriated shaped ground truth signal for logging

        Args:
            Y ([type]): [description]
            YType ([type]): [description]

        Returns:
            [type]: [description]
        """
        if Y is not None:
            isOkY = getIsOk(Y, self.missing_marker)
        else:
            YTrue = Y
        if YType == "cat":
            yDist = None
            YLossFuncs = [
                masked_CategoricalCrossentropy(self.missing_marker),
            ]
            if Y is not None:
                YClasses = np.unique(Y[:, np.all(isOkY, axis=0)])
                YTrue = np.ones((Y.shape[1], Y.shape[0], len(YClasses)), dtype=int) * (
                    int(self.missing_marker) if self.missing_marker is not None else -1
                )
                for yi in range(Y.shape[0]):
                    YTrueThis = get_one_hot(
                        Y[yi, isOkY[yi, :]][:, np.newaxis], len(YClasses)
                    )
                    YTrue[isOkY[yi, :], yi, :] = YTrueThis[:, 0, :]
        elif YType == "count_process":
            yDist = "poisson"
            YLossFuncs = [masked_PoissonLL_loss(self.missing_marker)]
            if Y is not None:
                YTrue = Y.T
        else:
            yDist = None
            YLossFuncs = [
                masked_mse(self.missing_marker),
                masked_R2(self.missing_marker),
                masked_CC(self.missing_marker),
            ]
            if Y is not None:
                YTrue = Y.T
        return YLossFuncs, YTrue, yDist