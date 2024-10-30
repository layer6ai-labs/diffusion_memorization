import inspect
import math
import numbers
from contextlib import contextmanager
from typing import Callable, Literal

import torch
from tqdm import tqdm

# A threshold for the dimension of the data, if the dimension is above this threshold, the hutchinson method is used
HUTCHINSON_DATA_DIM_THRESHOLD = 3500


def _jvp_mode(flag: bool, device: torch.device):
    """
    Flags that need to be set for jvp to work with attention layers.

    NOTE: This has been tested on torch version 2.1.1, hopefully,
    this issue will be resolved in a future version of torch
    as jvp mode reduces the speed of JVP computation.
    """
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(not flag)
        torch.backends.cuda.enable_mem_efficient_sdp(not flag)
        torch.backends.cuda.enable_math_sdp(flag)


@contextmanager
def _jvp_mode_enabled(device: torch.device):
    _jvp_mode(True, device)
    try:
        yield
    finally:
        _jvp_mode(False, device)
        
def compute_trace_of_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    method: Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None = None,
    hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
    chunk_size: int = 128,
    seed: int = 42,
    verbose: bool = False,
):
    """
    fn is a function mapping \R^d to \R^d, this function computes the trace of the Jacobian of fn at x.

    To do so, there are different methods implemented:

    1. The Hutchinson estimator:
        This is a stochastic estimator that uses random vector to estimate the trace.
        These random vectors can either come from the Gaussian distribution (if method=`hutchinson_gaussian` is specified)
        or from the Rademacher distribution (if method=`hutchinson_rademacher` is specified).
    2. The deterministic method:
        This is not an estimator and computes the trace by taking all the x.dim() canonical basis vectors times $\sqrt{d}$ (?)
        and taking the average of their quadratic forms. For data with small dimension, the deterministic method
        is the best.

    The implementation of all of these is as follows:
        A set of vectors of the same dimension as data are sampled and the value [v^T \\nabla_x v^T fn(x)] is
        computed using jvp. Finally, all of these values are averaged.

    Args:
        fn (Callable[[torch.Tensor], torch.Tensor]):
            A function that takes in a tensor of size [batch_size, *data_shape] and returns a tensor of size [batch_size, *data_shape]
        x (torch.Tensor): a batch of inputs [batch_size, input_dim]
        method (str, optional):
            chooses between the types of methods to evaluate trace.
            it defaults to None, in which case the most appropriate method is chosen based on the dimension of the data.
        hutchinson_sample_count (int):
            The number of samples for the stochastic methods, if deterministic is chosen, this is ignored.
        chunk_size (int):
            Jacobian vector products can be done in parallel for better speed-up, this is the size of the parallel batch.
    Returns:
        traces (torch.Tensor): A tensor of size [batch_size,] where traces[i] is the trace computed for the i'th batch of data
    """
    # use seed to make sure that the same random vectors are used for the same data
    # NOTE: maybe creating a fork of the random number generator is a better idea here!
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # save batch size and data dimension and shape
        batch_size = x.shape[0]
        data_shape = x.shape[1:]
        ambient_dim = x.numel() // x.shape[0]
        if ambient_dim > HUTCHINSON_DATA_DIM_THRESHOLD:
            method = method or "hutchinson_gaussian"
        else:
            method = method or "deterministic"

        # The general implementation is to compute the quadratic forms of [v^T \\nabla_x v^T score(x, t)] in a list and then take the average
        all_quadratic_forms = []
        sample_count = hutchinson_sample_count if method != "deterministic" else ambient_dim
        # all_v is a tensor of size [batch_size * sample_count, *data_shape] where each row is an appropriate vector for the quadratic forms
        if method == "hutchinson_gaussian":
            all_v = torch.randn(size=(batch_size * sample_count, *data_shape)).cpu().to(dtype=x.dtype)
        elif method == "hutchinson_rademacher":
            all_v = (
                torch.randint(size=(batch_size * sample_count, *data_shape), low=0, high=2)
                .cpu()
                .to(dtype=x.dtype)
                * 2
                - 1.0
            )
        elif method == "deterministic":
            all_v = torch.eye(ambient_dim).cpu().to(dtype=x.dtype) * math.sqrt(ambient_dim)
            # the canonical basis vectors times sqrt(d) the sqrt(d) coefficient is applied so that when the
            # quadratic form is computed, the average of the quadratic forms is the trace rather than their sum
            all_v = all_v.repeat_interleave(batch_size, dim=0).reshape(
                (batch_size * sample_count, *data_shape)
            )
        else:
            raise ValueError(f"Method {method} for trace computation not defined!")
        # x is also duplicated as much as needed for the computation
        all_x = (
            x.cpu()
            .unsqueeze(0)
            .repeat(sample_count, *[1 for _ in range(x.dim())])
            .reshape(batch_size * sample_count, *data_shape)
        )

        all_quadratic_forms = []
        rng = list(zip(all_v.split(chunk_size), all_x.split(chunk_size)))
        # compute chunks separately
        rng = tqdm(rng, desc="Computing the quadratic forms") if verbose else rng
        idx_dbg = 0
            
        with _jvp_mode_enabled(x.device), torch.amp.autocast("cuda"):
            for vx in rng:
                idx_dbg += 1

                v_batch, x_batch = vx
                v_batch = v_batch.to(x.device)
                x_batch = x_batch.to(x.device)
                all_quadratic_forms.append(
                    torch.sum(
                        v_batch * torch.func.jvp(fn, (x_batch,), tangents=(v_batch,))[1].to(x.dtype),
                        dim=tuple(range(1, x.dim())),
                    )
                )
    # concatenate all the chunks
    all_quadratic_forms = torch.cat(all_quadratic_forms)
    # reshape so that the quadratic forms are separated by batch
    all_quadratic_forms = all_quadratic_forms.reshape((sample_count, x.shape[0]))
    # take the average of the quadratic forms for each batch
    return all_quadratic_forms.mean(dim=0).to(x.device)
