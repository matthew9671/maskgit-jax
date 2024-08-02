import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial

# Convenient functions
from jax.nn import softmax

# Typing
from typing import Any, Callable, Sequence, Iterable, Dict, NamedTuple
from chex import Array, Scalar, ArrayTree
from numpy import False_

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any

def random_flip_batch(key, batch):
    N = batch.shape[0]
    flip = jnp.array(jr.bernoulli(key, .5, shape=(N,)), dtype=int).reshape((-1, 1, 1, 1))
    return flip * batch[:,:,::-1,:] + (1-flip) * batch

def compute_backward(y, t, model, params, config):
    y = y.flatten()
    D = y.shape[0]
    S = model.state_size
    forward_process = config["forward_process"]
    min_t = config["min_t"]
    eps = config["eps"]

    qt0 = forward_process.transition(t)
    # R^d_t(*1,*2): (S, S) float array of instantenous transition rates
    # for a single dimension
    Rt = forward_process.rate(t)
    Rt_eval_y = Rt[:, y].T
    x0_logits = model.apply(params, y, t, training=False)

    # p^{*1}_{0|t}(*2|y): (D, S) float array of marginal likelihoods for each dimension predicted by the model
    p0t_eval_y = softmax(x0_logits, axis=-1)
    # q^{*1}_{t|0}(y^d|*2): (D, S) float array of transition probabilities to y
    qt0_eval_y = qt0[:,y].T + eps
    st_eval_y = jnp.einsum("0x,d0->dx", qt0, p0t_eval_y / qt0_eval_y, 
                           precision=jax.lax.Precision.HIGHEST)

    # (D, S) float array that masks out y[d] for each d index
    y_mask = jnp.ones((D, S))
    y_mask = y_mask.at[jnp.arange(D), y].set(0.0)

    results = {
        "score": st_eval_y,
        "rates": (st_eval_y * Rt_eval_y) * y_mask,
        "x0_logits": x0_logits,
        "Rt": Rt,
    }

    return results

# The right thing to do here for jax is definitely just not worry about the batch dimension
def discrete_diffusion_loss_single(key, data, model, params, config):
    """
    key: a jax PRNGKey.
    data: (H, W, C) int array, each value should be in [0, S)
    """
    x0 = data.flatten()
    S = config["state_size"]
    D = x0.shape[0]
    forward_process = config["forward_process"]
    min_t = config["min_t"]
    eps = config["eps"]

    key_dropout, key_t, key_T, key_y = jr.split(key, 4)

    # --------------------------------------------------------------
    # 1. Sample a random t
    # --------------------------------------------------------------

    # TODO: sampling uniformly might not be a good idea
    # ofcourse, sampling non-uniformly is equivalent to changing the scalar rate parameter
    t = jr.uniform(key_t, minval=config["min_t"], maxval=config["max_t"])

    # q^d_{t|0}(*2|*1): (S, S) float array of finite-time transition probabilities
    # for a single dimension
    qt0 = forward_process.transition(t)
    # R^d_t(*1,*2): (S, S) float array of instantenous transition rates
    # for a single dimension
    Rt = forward_process.rate(t)

    # --------------------------------------------------------------
    # 2. Sample y from q(x_t | x_0)
    # --------------------------------------------------------------

    # q^{*1}_{t|0}(*2|x0): (D, S) float array of probabilities
    qt0_eval_x0 = qt0[x0, :]
    log_qt0_eval_x0 = jnp.log(qt0_eval_x0 + eps)
    # (D,) int array
    y = jr.categorical(key_y, logits=log_qt0_eval_x0)

    # --------------------------------------------------------------
    # 3. Evaluate the likelihood ratio predicted by the model
    # --------------------------------------------------------------

    x0_logits = model.apply(params, y, t, rngs={"dropout": key_dropout})

    # Assuming our model auto-adds a batch dimension, we want to remove it here:
    x0_logits = x0_logits[0]
    
    # p^{*1}_{0|t}(*2|y): (D, S) float array of marginal likelihoods for each dimension predicted by the model
    p0t_eval_y = softmax(x0_logits, axis=-1)
    # q^{*1}_{t|0}(y^d|*2): (D, S) float array of transition probabilities to y
    qt0_eval_y = qt0[:,y].T + eps

    # s_t^{\theta}^{*1}(y, *2): (D, S) float array of marginal likelihood ratios predicted by the model
    # Also known as the "concrete score" in (Lou et al. 2023)
#     st_eval_y = (p0t_eval_y / qt0_eval_y) @ qt0 
    st_eval_y = jnp.einsum("0x,d0->dx", qt0, p0t_eval_y / qt0_eval_y, 
                           precision=jax.lax.Precision.HIGHEST)

    # -------------------------------------------------------------
    # 4. Evaluate the likelihood ratios at t conditioned on data
    # -------------------------------------------------------------

    # q_{t|0}^{*1}(y^d, x_0^d): (D,) float array of sample probabilities in the forward process
    qt0_eval_y_x0 = qt0_eval_x0[jnp.arange(D), y] + eps
    # The likelihood ratio
    qt0_x_over_y = qt0_eval_x0 / qt0_eval_y_x0[:, None]

    # -------------------------------------------------------------
    # 5. Tying it all together
    # -------------------------------------------------------------

    # R^{*1}_t(*2,y^d): (D, S) float array of transition rates to y
    # for each dimension
    # Rt_eval_y = Rt.at[:,y].get().T
    Rt_eval_y = Rt[:,y].T

    # (D, S) float array that masks out y[d] for each d index
    y_mask = jnp.ones((D, S))
    y_mask = y_mask.at[jnp.arange(D), y].set(0.0)

    # (D, S) float array, with each entry corresponding to a choice of (d, x^d)
    # the cases where x^d = y^d are removed via masking
    score_entropy = Rt_eval_y * y_mask * (st_eval_y - qt0_x_over_y * jnp.log(st_eval_y + eps))
    
    # Compute the cross entropy between prediction and data
    x0_one_hot = jax.nn.one_hot(x0, S)
    # TODO: these are log probabilities, and they can probably be computed better from the actual logits
    logits = jnp.log(p0t_eval_y + eps)
    x0_nll = - jnp.mean(x0_one_hot * logits)

    loss = jnp.mean(score_entropy) + config["nll_weight"] * x0_nll
    
    # Sample from q_T to estimate the elbo
    # (S,) float array of the logits of the stationary distribution
    pi_logits = forward_process.target_logits()
    xT = jr.categorical(key_T, logits=pi_logits, shape=(D,))
    log_pi_eval_xT = jnp.sum(pi_logits[xT])
    elbo = jnp.sum(- score_entropy + Rt_eval_y * y_mask) + log_pi_eval_xT

    loss_dict = {
        "loss": loss,
        "elbo": elbo / D,
        # "noisy_sample": y,
        # "score_entropy_array": score_entropy,
        "nll": x0_nll
    }

    return loss_dict

def diffusion_batch_loss(key: PRNGKey,
                         model: Any,
                         params: Any,
                         batch: Array,
                         itr: int = 0,
                         **config):
    """
    Stardard loss computation mapped over the batch dimension.
    """
    B = batch.shape[0]
    vmapped = vmap(partial(discrete_diffusion_loss_single,
                               model=model,
                               params=params,
                               config=config))
    
    if config.get("random_flips"):
        key, flip_key = jr.split(key)
        batch = random_flip_batch(flip_key, batch)
        
    result = vmapped(jr.split(key, B), batch)
    return jnp.mean(result["loss"]), result