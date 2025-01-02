# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Weight Programming Optimization implementation which is similar to the framework
reported in the following paper:

    C. Mackin, et al., "Optimised weight programming for analogue memory-based
        deep neural networks" 2022. https://www.nature.com/articles/s41467-022-31405-1.
"""

from datetime import datetime

from typing import (
    Type,
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    Any
)

from copy import deepcopy
import pickle
from math import ceil, isnan

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from torch import (
    Tensor,
    Size,
    zeros,
    max as torch_max,
    abs as torch_abs,
    sum as torch_sum,
    linspace,
    cat,
    from_numpy,
    sort,
)

import torch.nn.functional as F

from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
    DualPairConductanceConverter,
    NPairConductanceConverter,
    CustomPairConductanceConverter,
)

from aihwkit.simulator.configs import InferenceRPUConfig

from aihwkit.nn import AnalogLinear


def downsample_weight_distribution(weights: Tensor,
                                   shape: Size) -> Tensor:
    """Downsamples weight distribution via interpolation

    Params:
        weights: original tensor of weights
        shape: torch.Size object containing shape of desired downsampled matrix

    Returns:
        downsampled N-dimensional weight distribution that is representative of
        overall network weight distribution. Note: this function will also
        correctly upsample weights when shape.numel() > weights.numel()

    """
    # interpolation on flattened, sorted weights likely represents distribution better
    # randomly shuffle to re-introduce disorder after sorting
    return shuffle_weights(F.interpolate(sort(weights.flatten())[0].unsqueeze(0).unsqueeze(1),
                                         size=(shape.numel(),),
                                         mode='linear',         # interpolation type
                                         align_corners=True,    # keep boundary values
                                         # detach otherwise pickle of deepcopy won't work
                                         ).reshape(shape).detach())


def stop_criterion(intermediate_result: OptimizeResult) -> None:
    """Terminate weight programming optimization once strategy once
    (current loss - baseline loss) / baseline loss + loss margin is less
    than zero, where baseline loss is determined by g_converter_baseline. Current
    loss is determined by the current weight programming optimization strategy.

    Args:
        intermediate_result: a keyword parameter containing an OptimizeResult with
        attributes x and fun, the best solution found so far and the objective
        function, respectively. Note that the name of the parameter must be
        intermediate_result for the callback to be passed an OptimizeResult.

    Raises:
        StopIteration: when intermediate_result.fun (loss function) becomes
        negative, which means optimization has adequately converged as defined
        by the loss_margin parameter

    """

    print("\t(current loss - baseline loss) / baseline loss + loss margin = %0.10f (%s)"
          % (intermediate_result.fun, datetime.now()))

    if intermediate_result.fun < 0.0:
        raise StopIteration


def partition_parameters(x: Tensor, d: Dict) -> Tuple:
    """Separates array of x parameters from differential evolution into
    parameters corresponding to f_p factors (if they exist) and x_g parameter,
    which will be used to find corresponding valid conductance combinations.

    Args:
        x: hypercube parameters corresponding to a weight programming strategy
        d: dictionary with all WeightProgrammingOptimization attribution
            that were previously serialized/pickled to be compatible with
            scipy differential evolution

    Returns:
        f_lst: list of conductance pair scaling parameters f
        k_w: unitless to micoSiemens weight rescaling factor [uS/1]
        x_g: hypercube parameters corresponding to remaining weight programming
            strategy parameters
    """
    f_lst = d['f_lst'].copy()                                       # don't alter original
    cnts = f_lst.count(None)                                        # count None occurences
    if cnts > 0:                                                    # only compute if None exists
        inds = [i for i, f in enumerate(f_lst) if f is None]        # indices of None occurences
        f_lst_none = [x_f * (d['f_max'] - d['f_min']) + d['f_min']  # calc f vals to replace None
                      for x_f in x[0:cnts]]
        for ind, f_val in zip(inds, f_lst_none):                    # replace None values in f_lst
            f_lst[ind] = f_val
        x = x[cnts:]                                                # done - can drop params
    x_k_w = 1.0 if d['use_max_kw'] else x[0]                        # get xkw
    x_g = x if d['use_max_kw'] else x[1:]                           # get remaining xg values

    w_us_absolute_max = sum(f * (d['g_max'] - d['g_min']) for f in f_lst)
    lower_bound, upper_bound = 0.05 * w_us_absolute_max, w_us_absolute_max
    w_us_max = x_k_w * (upper_bound - lower_bound) + lower_bound
    k_w = w_us_max / d['max_abs_weight_unitless']

    return (f_lst, k_w, x_g)


def reformat_x_g(x_g: Tensor, len_f_lst: int, symmetric: bool) -> Tuple:
    """Get x values in same format as CustomPairConductanceConverter g_lst

    Args:
        x_g: list of hypercube parameters
        len_f_lst: length of conductance pair scaling parameters f in f_lst
        symmetric: whether or not weight programming optimization solution
            will be symmetric for positive and negative weights (reduces
            dimensionality of optimization problem)

    Returns:
        x_g_lst: reformatted xg to correspond to CustomPairConductanceConverter
            g_lst formatting
        g_len: number discretized weights specified in CustomPairConductanceConverter
    """
    g_len = int(len(x_g) / (2 * len_f_lst - 1))
    x_g_lst = [x_g[i:i + g_len].tolist() for i in range(0, len(x_g), g_len)]
    g_len = g_len * 2 - 1 if symmetric else g_len
    return x_g_lst, g_len


def loss_weights(model: AnalogLinear, t_steps: List[float], test_weights: Tensor,
                 max_abs_weight_unitless: float, loss_baseline: float,
                 loss_margin: float, get_baseline: bool = False) -> float:
    """Computes Time-Averaged Normalized Mean Squared Error (TNMSE) for weight distribution

    Args:
        model: one AnalogLinear layer used for test evaluation
        t_steps: time steps to optimize over
        test_weights: test weights (2d) that we want to implement
        max_abs_weight_unitless: maximum weight value positive or negative
        loss_baseline: baseline loss (tnmse) for baseline weight programming strategy
        loss_margin: how much we are trying to beat the baseline loss by 0.1 = 10%
        get_baseline: whether to return true loss (tnmse) or the normalized version where
            we have beat the baseline loss by the loss margin amount once the value becomes
            less than zero

    Returns:
        tnmse: Time-Averaged Normalized Mean Squared Error (TNMSE), usually the normalized
            version where where we have beat the baseline loss by the loss margin when
            the value becomes less than zero.

    """
    error_amplification = 10.

    # could optionally reshuffle weights at each time step for better representation (but slower)
    for layer in model.analog_layers():
        layer.set_weights(test_weights.to(test_weights.device).T.clone(),
                          zeros(model.out_features).to(test_weights.device))

    nse = 0
    for t in t_steps:
        model.program_analog_weights()  # new prog errors + new drift coefficients each time
        model.drift_analog_weights(t)   # drifts weights + new read noise + new drift comp alpha
        effective_weights_lst = []
        for layer in model.analog_layers():     # only one layer
            for tile in layer.analog_tiles():   # potentially split across multiple tiles
                effective_weights_lst.append(
                    tile.alpha * tile.get_weights()[0].T)   # global and channelwise
        effective_weights = cat(effective_weights_lst, 0)
        nse += torch_sum((error_amplification * (effective_weights.to(test_weights.device)
                                                 - test_weights) / max_abs_weight_unitless) ** 2)
    tnmse = nse / (model.in_features * model.out_features * len(t_steps))
    tnmse = tnmse.detach().cpu().numpy()
    tnmse = np.inf if isnan(tnmse) else tnmse

    return tnmse if get_baseline else (tnmse - loss_baseline) / loss_baseline + loss_margin


def shuffle_weights(weights: Tensor) -> Tensor:
    """Sample weights to test programming strategy

    Params:
        weights: tensor of weights

    Returns:
        shuffled tensor of weights with equivalent dimensions

    """
    # numpy shuffle implementation (faster than torch version)
    weights_np = weights.detach().cpu().numpy()     # ensure correct shape
    np.random.shuffle(weights_np)                   # returns none, shuffle in place
    return from_numpy(weights_np)                   # back to torch


def loss_rpu_config(d: Dict) -> float:
    """Computes Time-averaged Normalized Mean Squared Error (TNMSE)
    between the target weight distribution and implemented weight
    distribution according to weight programming strategy and
    corresponding programming errors, read noise, drift, and
    drift compensation. This serves as a loss function to be
    minimized.

    Args:
        d: dictionary with all WeightProgrammingOptimization attribution
            that were previously serialized/pickled to be compatible with
            scipy differential evolution
    Returns:
        Loss, which is a Time-averaged Normalized Mean Squared Error (TNMSE)
    """

    device = d['device']    # must be cpu for scipy differential evoluation multiple workers

    # create test model
    model = AnalogLinear(d['test_in_features'],
                         d['test_out_features'],
                         bias=False,
                         rpu_config=deepcopy(d['rpu_config'])).eval().to(device)

    return loss_weights(deepcopy(model),
                        d['t_steps'],
                        d['weights_downsampled_unitless'],
                        d['max_abs_weight_unitless'],
                        d['loss_baseline'],
                        d['loss_margin'],
                        get_baseline=d['get_baseline'])


def span_of_remaining_pairs(span_of_each_pair: List, ind: int) -> Tensor:
    """Computes the span of the remaining conductance pairs. Informs interdependent
    constraints on conductance programming based on how previous conductance pair was
    programmed.

    Args:
        span_of_each_pair: range of each conductance pair including f factor
        ind: index of remaining conductance pairs

    Returns:
        Remaining conductance range (i.e. maximum positive/negative conductance
        value that could programmed in the remaining conductance pairs)
    """
    remaining_span = 0 if ind > len(span_of_each_pair) else sum(span_of_each_pair[ind:])
    return remaining_span


def denormalize(x: Tensor, d: Dict) -> Tuple:
    """De-normalizes a hypercube input parameter x to corresponding
    f_p values and gp_p and gm_p values.

    Args:
        x: hypercube parameters representing a weight programming strategy
        d: dictionary with all WeightProgrammingOptimization attribution
            that were previously serialized/pickled to be compatible with
            scipy differential evolution

    Returns:
        f_lst: conductance pair scaling factors for CustomPairConductanceConverter
        g_lst: conductance programming spec for CustomPairConductanceConverter
    """
    # pylint: disable-msg=too-many-locals

    f_lst, k_w, x_g = partition_parameters(x, d)
    w_discretized_us = (k_w * d['weights_discretized_unitless']).tolist()    # uS weights
    span_of_each_pair = [f * (d['g_max'] - d['g_min']) for f in f_lst]
    xg_lst, g_len = reformat_x_g(x_g, len(f_lst), d['symmetric'])

    if d['symmetric']:
        w_discretized_us = w_discretized_us[0:int(g_len / 2) + 1]

    # solve for g_lst
    g_lst = [[None for _ in range(g_len)] for _ in range(2 * len(f_lst))]
    for i, w_us in enumerate(w_discretized_us):
        w_us_center = w_us
        for j, f_factor in enumerate(f_lst):

            # x to determine gp and gm breakdown that produces delta_g
            x_g = xg_lst[2 * j][i]

            # x to determine conductance pair contribution: delta_g = g_p - g_m
            # (i.e. which equipotential line)
            x_delta_g = xg_lst[2 * j + 1][i] if j < len(f_lst) - 1 else 1.0

            span_of_current = f_factor * (d['g_max'] - d['g_min'])
            lower_bound = max(w_us_center - span_of_remaining_pairs(span_of_each_pair, j + 1),
                              -span_of_current)
            upper_bound = min(w_us_center + span_of_remaining_pairs(span_of_each_pair, j + 1),
                              span_of_current)

            # delta_g (which equipotential line) for this g pair
            f_delta_g = x_delta_g * (upper_bound - lower_bound) + lower_bound

            w_us_center = w_us_center - f_delta_g   # new center based on delta_g from previous
            delta_g = f_delta_g / f_factor          # remove f factor amplification

            # gp lower bound for equipotnl line + underprog
            lower_bound = d['g_min'] + max(delta_g, 0) - d['g_err']

            # gp upper bound for equipotnl line + overprog
            upper_bound = d['g_max'] + min(delta_g, 0) + d['g_err']

            g_p = x_g * (upper_bound - lower_bound) + lower_bound   # where on equipotential line
            g_p = max(min(g_p, d['g_max']), d['g_min'])   # constrain (for slight over/underprog)
            g_m = g_p - delta_g
            g_m = max(min(g_m, d['g_max']), d['g_min'])   # constrain (for slight over/underprog)

            g_lst[2 * j][i] = g_p
            g_lst[2 * j + 1][i] = g_m

            if d['symmetric']:
                g_lst[2 * j][g_len - i - 1] = g_m          # mirror for positive weights
                g_lst[2 * j + 1][g_len - i - 1] = g_p

    return (f_lst, g_lst)


def loss_fxn(x: Tensor, *args: Union[bytes, bytearray]) -> float:
    """Computes loss based on hypercube x values being probed by differential evolution.

    Args:
        x: hypercube values corresponding to a programming strategy
        args: any additional arguments necessary for optimization,
            must be pickled to work with multiple workers in
            scipy differential evolution algorithm

    Returns:
        Loss of corresponding programming strategy defined by x
    """
    d = deepcopy(pickle.loads(args[0]))
    f_lst, g_lst = denormalize(x.copy(), d)

    # udpate rpu_config g_converter
    rpu_config = deepcopy(d['rpu_config'])
    rpu_config.noise_model.g_converter \
        = CustomPairConductanceConverter(f_lst.copy(),
                                         g_lst.copy(),
                                         g_min=d['g_min'],
                                         g_max=d['g_max'],
                                         invertibility_test=False)
    d.update({'rpu_config': rpu_config})    # update rpu_config
    iterations = 1
    # input deepcopy(d) so each run gets independent copy + optional averaging
    return sum(loss_rpu_config(deepcopy(d)) for _ in range(iterations)) / iterations


class WeightProgrammingOptimizer:
    """Weight Programming Optimization Class.

    Uses differential evoluation to minimize the time-averaged normalized mean-squared
    error (TNMSE) loss between ideal weights and effective weights, which include device
    non-idealities such as programming errors, read noise, conductance drift, and
    algorithmic drift compensation. Can also optimize for significance pair scaling
    factors f and be employed to optimize symmetric weight programming (for negative
    and positive weights), which reduces the dimensionality and search space. Alternatively,
    one can also optimize weight programming in a ~2x higher-dimensional space for positive
    and negative weights in the event the weight distribution is highly asymmetric.

    Params:
        weights: ideal unitless weight distribution to program
        f_lst: list significance pair scaling factors. Passing a list of None values
            will cause optimizer to solve for the best hardware f factors. Alternatively,
            a specified f_lst such as [1.0, 3.0] will constrain the optimization to a
            specific set of hardware f factors.
        rpu_config: resistive processing unit configuration.
        t_steps: time steps used in the optimization process. Will try to minimize weight
            errors at all of the specified time steps.
        g_converter_baseline: a baseline g_converter which the weight programming optimizer
            will try to outperform. You can input an instantiated SinglePairConductanceConverter
            or a DualPairConductanceConverter. Alternatively, you can also pass a previously
            optimized CustomPairConductanceConverter in the event you would like to improve
            on an previously optimized weight programming strategy.
        symmetric: boolean that specifies whether or not to employ a symmetric weight programming
            strategy for negative and positive weights. Enforcing symmetric reduces the
            optimization dimensionality / search space and leads to faster optimization times.
        kwargs: optional parameters that allows the user to override the baseline parameters
            passed to scipy differential_evolution algorithm. The baseline parameters have been
            heavily optimized and should be sufficient. In some cases, it may be beneficial to
            adjust these values to improve the speed or quality of results.

    Returns:
        CustomPairConductanceConverter: instantiated with optimized weight programming strategy
        success: boolean flag indicating whether scipy differential_evolution successfully
            terminated

    """
    # pylint: disable=too-many-instance-attributes, too-many-statements
    def __init__(
            self,
            weights: Tensor,
            f_lst: List,
            rpu_config: Type[InferenceRPUConfig],
            t_steps: List,
            g_converter_baseline: Union[SinglePairConductanceConverter,
                                        DualPairConductanceConverter,
                                        CustomPairConductanceConverter],
            symmetric: bool = True,
            **kwargs: Optional[Any]):

        self.f_lst = f_lst
        self.rpu_config = deepcopy(rpu_config)  # to avoid modifying outside

        # turn off out scales during optimization (to avoid weight distortion)
        self.mapping_saved = deepcopy(self.rpu_config.mapping)
        self.rpu_config.mapping.weight_scaling_omega = 0.0
        self.rpu_config.mapping.weight_scaling_columnwise = False
        self.rpu_config.mapping.learn_out_scaling = False
        self.rpu_config.mapping.out_scaling_columnwise = False

        self.noise_model = self.rpu_config.noise_model
        self.t_steps = t_steps

        self.symmetric = symmetric
        self.test_in_features = self.rpu_config.mapping.max_input_size

        # default settings
        self.nbins = 6                      # weight RHS (positive) number of bins
        self.test_out_features = 20         # weight/drift compensation averaging
        self.g_err = 0.00                   # over/under programming
        self.f_min = 0.1                    # min hardware f_p value
        self.f_max = 5.0                    # max hardware f_p value
        self.use_max_kw = False             # force max g range
        self.loss_margin = 0.125            # margin to beat baseline
        self.baseline_iterations = 100      # iterations to estimate average baseline loss
        self.callback = stop_criterion      # default to print convergence status updates
        self.disp = True                    # print status (disable for tests)

        # differential evolution params
        self.strategy = 'best1bin'          # best1bin works well
        self.popsize = 100                  # defaults to 100x number of dimensions
        self.maxiter = 200                  # max iterations
        self.tol = 0.0                      # tolerance
        self.atol = 0.02                    # absolute tolerance
        self.mutation = 0.1                 # mutation/dithering: larger = wider search space
        self.recombination = 0.3            # recombination/crossover: larger = faster convergence
        self.polish = True                  # polish with grad descent
        self.workers = -1                   # parallel, -1 = max CPUs available
        self.seed = 100                     # seed, reproducibility
        self.x_init = 'latinhypercube'      # x initialization method
        self.device = 'cpu'                 # must be cpu for diff evo multi-workers

        if set(kwargs.keys()).issubset(self.__dict__.keys()):
            self.__dict__.update(kwargs)
        else:
            invalid_kwargs = set(kwargs.keys()).difference(self.__dict__.keys())
            raise Exception("The following argument(s) are not supported:\n %s" % invalid_kwargs)

        valid_g_converter_baselines = [SinglePairConductanceConverter,
                                       DualPairConductanceConverter,
                                       NPairConductanceConverter,
                                       CustomPairConductanceConverter]

        if not any(isinstance(g_converter_baseline, g_converter)
                   for g_converter in valid_g_converter_baselines):
            raise TypeError("g_converter_baseline = %s not supported"
                            % type(g_converter_baseline).__name__)

        self.g_converter_baseline = deepcopy(g_converter_baseline)
        self.g_min = self.g_converter_baseline.g_min            # use baseline g_min, g_max
        self.g_max = self.g_converter_baseline.g_max            # for fair comparison
        self.g_err = self.g_err * (self.g_max - self.g_min)     # for over/underprog

        g_lst_init = [[self.g_min] * (self.nbins * 2 - 1)] * 2 * len(f_lst)
        self.g_converter = CustomPairConductanceConverter(f_lst,
                                                          g_lst_init,
                                                          g_max=self.g_max,
                                                          g_min=self.g_min,
                                                          invertibility_test=False,
                                                          )

        # other parameters
        self.get_baseline = False               # switch to get true loss or relative to baseline
        self.loss_baseline = float('inf')       # g_converter_baseline loss
        self.best_loss = float('inf')           # best loss after optimization

        self.weights_downsampled_unitless \
            = downsample_weight_distribution(weights.to(self.device).flatten(),
                                             Size([self.test_in_features, self.test_out_features]))

        self.max_abs_weight_unitless = torch_max(torch_abs(self.weights_downsampled_unitless))
        self.weights_discretized_unitless = linspace(-self.max_abs_weight_unitless,
                                                     self.max_abs_weight_unitless,
                                                     self.nbins * 2 - 1).to(self.device)

    def generate_hop_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """Generates the bounds applied to the differential weight evolution
        algorithm.

        Returns: tuple of tuples which specify the number and constraints
            (hypercube) used for differential evoluation optimization
        """
        g_bnds = (0.0, 1.0)

        n_bnds = len(self.weights_discretized_unitless.tolist())

        if self.symmetric:
            n_bnds = ceil(n_bnds / 2)

        n_bnds *= (2 * len(self.f_lst) - 1)                 # multiply by conductance pairs
        n_bnds += sum(f is None for f in self.f_lst)        # f_lst params if necessary
        n_bnds = n_bnds if self.use_max_kw else n_bnds + 1  # for kw
        bnds = (g_bnds,) * n_bnds
        return bnds

    def get_loss_baseline(self) -> float:
        """Estimates the time-averaged normalized mean square error (TNMSE) value
        for the weight progamming optimizer to beat based on a user-specified
        standard weight programming procedure.

        Returns:
            Loss, which is a time-averaged normalized mean square error (TNMSE),
            which the weight programming optimizer will try to outperform
        """

        # copy rpu_config
        rpu_config = deepcopy(self.rpu_config)

        # udpate rpu_config g_converter with baseline for comparison
        rpu_config.noise_model.g_converter = self.g_converter_baseline
        baseline_loss_params = deepcopy(self.__dict__)
        baseline_loss_params.update({'get_baseline': True})

        loss_baseline = sum(loss_rpu_config(deepcopy(baseline_loss_params))
                            for _ in range(self.baseline_iterations)) / self.baseline_iterations
        if self.disp:
            print("\n\tloss baseline = %f (%s)\n" % (loss_baseline,
                                                     type(self.g_converter_baseline).__name__))

        return loss_baseline

    def differential_weight_evolution(self) -> Tuple[CustomPairConductanceConverter, bool]:
        """Runs differential evolution for weight programming optimization.

        To increase the chances of finding a good minima: increase popsize
        along with mutation, but lower recombination.

        Returns:
            g_converter: CustomPairConductanceConverter instantiated
                with optimal f_lst, g_lst weight programming specifications
            success: whether weight programming optimization was successful
                or not
        """

        bnds = self.generate_hop_bounds()
        if self.disp:
            print("\t\thypercube dimensions = %d \n" % len(bnds))

        res = differential_evolution(loss_fxn,
                                     bnds,
                                     args=(pickle.dumps(deepcopy(self.__dict__)),),
                                     callback=self.callback,
                                     strategy=self.strategy,
                                     maxiter=self.maxiter,
                                     popsize=self.popsize,
                                     tol=self.tol,
                                     atol=self.atol,
                                     mutation=self.mutation,
                                     recombination=self.recombination,
                                     seed=self.seed,
                                     disp=False,
                                     polish=self.polish,
                                     workers=self.workers,
                                     updating='deferred' if self.workers == -1 else 'immediate',
                                     )

        # success if converged or beat baseline by margin
        success = True if res.fun < 0. else res.fun

        # invert loss in tnmse
        self.best_loss = (res.fun - self.loss_margin) * self.loss_baseline + self.loss_baseline

        f_lst, g_lst = denormalize(res.x.copy(), self.__dict__.copy())
        optimal_g_converter = CustomPairConductanceConverter(f_lst=f_lst,
                                                             g_lst=g_lst,
                                                             g_min=self.g_min,
                                                             g_max=self.g_max,
                                                             invertibility_test=False)

        return optimal_g_converter, success

    def run_optimizer(self) -> Tuple:
        """Runs a series of steps that optimizes the weight programming
        strategy based on programming noise, read noise, conductance-
        dependent drift models, and drift compensation so as to maintain
        weight fidelity as best as possible overtime and help the
        network achieve iso-accuracy.

        Returns:
            optimal_g_converter: optimal weight programming strategy in
                the form of an instantiated CustomPairConductanceConverter
            success: boolean flag specifying whether or not the optimization
                was a success

        """

        self.loss_baseline = self.get_loss_baseline()
        optimal_g_converter, success = self.differential_weight_evolution()

        # update object
        self.rpu_config.mapping = self.mapping_saved  # restore mapping params
        self.rpu_config.noise_model.g_converter = optimal_g_converter
        self.f_lst = self.rpu_config.noise_model.g_converter.f_lst

        return (optimal_g_converter, success)
