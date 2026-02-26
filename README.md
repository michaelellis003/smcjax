# smcjax

[![CI](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10|3.11|3.12|3.13-blue)](https://www.python.org)
[![License](https://img.shields.io/github/license/michaelellis003/smcjax)](https://github.com/michaelellis003/smcjax/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/Pyright-enabled-brightgreen)](https://github.com/microsoft/pyright)

Sequential Monte Carlo and particle filtering in JAX.

**smcjax** provides JIT-compiled, GPU-ready particle filters.

## Features

- **Bootstrap (SIR) particle filter** via `jax.lax.scan`
- **4 resampling schemes** (via [Blackjax](https://github.com/blackjax-devs/blackjax)): systematic, stratified, multinomial, residual
- **Effective sample size** (ESS) computation (via Blackjax)
- **Conditional resampling** with configurable ESS threshold
- All functions are `jit`- and `vmap`-compatible
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
    __init__.py          # Public API (re-exports blackjax ESS & resampling)
    types.py             # PRNGKeyT, Scalar (matches Dynamax)
    containers.py        # ParticleState, ParticleFilterPosterior
    weights.py           # log_normalize, normalize
    bootstrap.py         # Bootstrap (SIR) particle filter
```

ESS and resampling (systematic, stratified, multinomial, residual) are
provided by [Blackjax](https://github.com/blackjax-devs/blackjax) and
re-exported from `smcjax` for convenience.

## Cross-Validation

The bootstrap filter is tested against reference libraries:

| Module | Reference | Validation |
|--------|-----------|------------|
| `bootstrap` | [Dynamax](https://github.com/probml/dynamax) Kalman filter | Log-ML within 5% of exact |
| `bootstrap` | [particles](https://github.com/nchopin/particles) (Chopin) | Log-ML within 3 nats |

## Roadmap

| Phase | What |
|-------|------|
| 1 (current) | Bootstrap particle filter |
| 2 | Auxiliary particle filter |
| 3 | EKF/UKF proposal particle filters |
| 4 | Liu-West filter, PMMH |

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
