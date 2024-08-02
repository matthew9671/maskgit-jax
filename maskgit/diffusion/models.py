import jax
import jax.numpy as jnp
import numpy as np

# Typing
from typing import Any, Callable, Sequence, Iterable, Dict, NamedTuple
from chex import Array, Scalar

from jax.scipy.special import logsumexp

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any

class UniformRate():
    def __init__(self, config):
        self.state_size = S = config["state_size"]
        self.scalar_rate = config["scalar_rate"]

        rate = self.scalar_rate * jnp.ones((S, S))
        rate -= jnp.diag(jnp.diag(rate))
        rate -= jnp.diag(jnp.sum(rate, axis=1))
        self.eigvals, self.eigvecs = jnp.linalg.eigh(rate)
        self.rate_matrix = rate

    def target_logits(self) -> Array:
        S = self.state_size
        return - jnp.ones((S,)) * jnp.log(S)
        
    def rate(self, t: Scalar) -> Array:
        return self.rate_matrix

    def transition(self, t: Scalar, t0: Scalar = 0) -> Array:
        trans = self.eigvecs @ jnp.diag(jnp.exp(self.eigvals * t)) @ self.eigvecs.T
        trans = jnp.clip(trans, 0., 1.)
        return trans
    
class GaussianTargetRate():
    def __init__(self, config):
        self.state_size = S = config["state_size"]
        
        self.rate_sigma = config["rate_sigma"]
        self.Q_sigma = config["Q_sigma"]
        self.time_exponential = config["time_exponential"]
        self.time_base = config["time_base"]

        rate = np.zeros((S,S))

        vals = np.exp(-np.arange(0, S)**2/(self.rate_sigma**2))
        for i in range(S):
            for j in range(S):
                if i < S//2:
                    if j > i and j < S-i:
                        rate[i, j] = vals[j-i-1]
                elif i > S//2:
                    if j < i and j > -i+S-1:
                        rate[i, j] = vals[i-j-1]
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * np.exp(- ( (j+1)**2 - (i+1)**2 + S*(i+1) - S*(j+1) ) / (2 * self.Q_sigma**2)  )
        
        rate -= np.diag(np.diag(rate))
        rate -= np.diag(np.sum(rate, axis=1))
        self.eigvals, self.eigvecs = np.linalg.eig(rate)
        self.inv_eigvecs = np.linalg.inv(self.eigvecs)
        self.base_rate = jnp.array(rate, dtype=np.float32)
        self.eigvals = jnp.array(self.eigvals, dtype=np.float32)
        self.eigvecs = jnp.array(self.eigvecs, dtype=np.float32)
        self.inv_eigvecs = jnp.array(self.inv_eigvecs, dtype=np.float32)
    
    def _integral_rate_scalar(self, t):
        return self.time_base * (self.time_exponential ** t) - self.time_base
    
    def _rate_scalar(self, t):
        return self.time_base * np.log(self.time_exponential) * (self.time_exponential ** t)

    def target_logits(self):
        S = self.state_size
        initial_dist_std = self.Q_sigma
        logits = - ((jnp.arange(1, S+1) - S//2)**2) / (2 * initial_dist_std**2)
        return logits - logsumexp(logits)
    
    def rate(self, t):
        S = self.state_size
        rate_scalar = self._rate_scalar(t)

        return self.base_rate * rate_scalar

    def transition(self, t, t0=0):
        S = self.state_size

        integral_rate_scalar = self._integral_rate_scalar(t+t0) - self._integral_rate_scalar(t0)

        adj_eigvals = integral_rate_scalar * self.eigvals

        trans = jnp.einsum("ij,jk,kl->il", self.eigvecs, jnp.diag(jnp.exp(adj_eigvals)), self.inv_eigvecs, 
                           precision=jax.lax.Precision.HIGHEST)
        trans = jnp.clip(trans, 0., 1.)

        return trans
    
class AbsorbingRate():
    def __init__(self, config):
        self.state_size = S = config["state_size"]
        self.scalar_rate = 1
        self.eps = config["rate_eps"]
        
        mask = S-1

        rate = np.zeros((S, S))
        rate[:-1, -1] = self.scalar_rate
        rate -= np.diag(jnp.sum(rate, axis=1))
        self.eigvals, self.eigvecs = jnp.linalg.eigh(rate)
        self.eigvals, self.eigvecs = np.linalg.eig(rate)
        self.inv_eigvecs = np.linalg.inv(self.eigvecs)
        
        self.base_rate = jnp.array(rate, dtype=np.float32)
#         self.rate_matrix = self.base_rate
        self.eigvals = jnp.array(self.eigvals, dtype=np.float32)
        self.eigvecs = jnp.array(self.eigvecs, dtype=np.float32)
        self.inv_eigvecs = jnp.array(self.inv_eigvecs, dtype=np.float32)
        
    def target_logits(self):
        S = self.state_size
        logits = - jnp.ones((S,)) * 10000
        return logits.at[-1].set(0)
        
    def _integral_rate_scalar(self, t):
        return -jnp.log1p(-(1 - self.eps) * t)
    
    def _rate_scalar(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)
        
    def rate(self, t):
        return self._rate_scalar(t) * self.base_rate

    def transition(self, t, t0 = 0):
        S = self.state_size

        integral_rate_scalar = self._integral_rate_scalar(t+t0) - self._integral_rate_scalar(t0)

        adj_eigvals = integral_rate_scalar * self.eigvals

        trans = jnp.einsum("ij,jk,kl->il", self.eigvecs, jnp.diag(jnp.exp(adj_eigvals)), self.inv_eigvecs, 
                           precision=jax.lax.Precision.HIGHEST)
        trans = jnp.clip(trans, 0., 1.)

        return trans