# GPT2 on JAX

Implements a training loop for GPT2 in JAX. I wanted to learn more about modelling
and implement a training loop from scratch.

The latest library versions aren't used since I'm using an M1 MacBook Pro as my
development machine. Apparently versions of `jaxlib>0.5.0` are broken such that
XLA emits Stable HLO which is incompatible with Metal.
(see https://github.com/jax-ml/jax/issues/27146). To take advantage of JAX's JIT
compiler for hardware accelerated matmuls on macOS, we unfortunately need to
live with this situation.

Future work (probably in another branch) will include implementing an inference
branch for this GPT implementation to make the trained parameters actually useful.

## Usage

1. Use `pipenv` to install dependencies form `Pipfile` (e.g. `pipenv install`)
2. Download a big text file to use as training data
3. Modify the argument to `data.load_text()` in `trainer.py` to be the path to
   your source data
4. Tune hyperparameters
5. Enter the Python environment with the installed dependencies (e.g. `pipenv shell`) 
   and run `python trainer.py`