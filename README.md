# smcjax

[![CI](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10|3.11|3.12|3.13-blue)](https://www.python.org)
[![License](https://img.shields.io/github/license/michaelellis003/smcjax)](https://github.com/michaelellis003/smcjax/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/Pyright-enabled-brightgreen)](https://github.com/microsoft/pyright)

Sequential Monte Carlo and particle filtering in JAX.

**smcjax** provides JIT-compiled, GPU-ready particle filters designed to
eventually integrate with [Dynamax](https://github.com/probml/dynamax)
(Kevin Murphy's JAX state space model library).

## Features

- **Bootstrap (SIR) particle filter** via `jax.lax.scan`
- **4 resampling schemes**: systematic, stratified, multinomial, residual
- **Effective sample size** (ESS) computation
- **Conditional resampling** with configurable ESS threshold
- All functions are `jit`- and `vmap`-compatible
- Interface-compatible with [Blackjax](https://github.com/blackjax-devs/blackjax) resampling
- Type annotations via [jaxtyping](https://github.com/google/jaxtyping)

## Installation

```bash
pip install smcjax
```

Or from source:

```bash
git clone https://github.com/michaelellis003/smcjax.git
cd smcjax
uv sync
```

## Quick Example

```python
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from smcjax import bootstrap_filter

# Define a 1-D linear Gaussian state space model
m0, P0 = jnp.array([0.0]), jnp.array([[1.0]])
F, Q = jnp.array([[0.9]]), jnp.array([[0.25]])
H, R = jnp.array([[1.0]]), jnp.array([[1.0]])

def initial_sampler(key, n):
    return tfd.MultivariateNormalFullCovariance(m0, P0).sample(n, seed=key)

def transition_sampler(key, state):
    mean = (F @ state[:, None]).squeeze(-1)
    return tfd.MultivariateNormalFullCovariance(mean, Q).sample(seed=key)

def log_observation_fn(emission, state):
    mean = (H @ state[:, None]).squeeze(-1)
    return tfd.MultivariateNormalFullCovariance(mean, R).log_prob(emission)

# Simulate some data
key = jr.PRNGKey(0)
T = 100
emissions = jr.normal(key, (T, 1))

# Run the bootstrap particle filter
posterior = bootstrap_filter(
    key=jr.PRNGKey(1),
    initial_sampler=initial_sampler,
    transition_sampler=transition_sampler,
    log_observation_fn=log_observation_fn,
    emissions=emissions,
    num_particles=1_000,
)

print(f"Log marginal likelihood: {posterior.marginal_loglik:.2f}")
print(f"Particles shape: {posterior.filtered_particles.shape}")
print(f"Mean ESS: {posterior.ess.mean():.1f}")
```

## Architecture

```
smcjax/
    __init__.py          # Public API
    types.py             # PRNGKeyT, Scalar (matches Dynamax)
    containers.py        # ParticleState, ParticleFilterPosterior
    weights.py           # log_normalize, normalize
    ess.py               # Effective sample size
    resampling.py        # systematic, stratified, multinomial, residual
    bootstrap.py         # Bootstrap (SIR) particle filter
```

## Cross-Validation

All implementations are tested against reference libraries:

| Module | Reference | Validation |
|--------|-----------|------------|
| `ess` | [Blackjax](https://github.com/blackjax-devs/blackjax) | Exact match |
| `resampling` | Blackjax | Identical indices with same PRNG key |
| `bootstrap` | [Dynamax](https://github.com/probml/dynamax) Kalman filter | Log-ML within 5% of exact |
| `bootstrap` | [particles](https://github.com/nchopin/particles) (Chopin) | Log-ML within 3 nats |

## Dynamax Contribution Roadmap

This library is being developed with the goal of contributing particle
filtering capabilities to Dynamax, which currently has zero PF
implementations and three open issues requesting them
([#112](https://github.com/probml/dynamax/issues/112),
[#272](https://github.com/probml/dynamax/issues/272),
[#275](https://github.com/probml/dynamax/issues/275)).

| Phase | What | Dynamax Issue |
|-------|------|---------------|
| 1 (current) | Bootstrap particle filter | #112 |
| 2 | Auxiliary particle filter | #112 |
| 3 | EKF/UKF proposal particle filters | #272 |
| 4 | Liu-West filter, PMMH | #275 |
| 5 | PR to Dynamax | All |

## Development

```bash
uv sync                          # Install all deps
uv run pre-commit install        # Set up pre-commit hooks
uv run pytest -v --cov           # Run tests with coverage
uv run ruff check . --fix        # Lint
uv run pyright                   # Type check
```

## License

Apache-2.0
