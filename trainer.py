from typing import Mapping, Union

from datetime import datetime
import dataclasses
import functools
import os

import flax.struct
import flax.linen as nn

import jax
import jax.numpy as jnp

import orbax.checkpoint as ocp

import optax

import data
import model


VariableName = str
VariableTreeNode = Union[Mapping[VariableName, "VariableTreeNode"], jax.Array]


NUM_EPOCHS = 1
STEPS_PER_EPOCH = 10

CONTEXT_LEN = 512


@flax.struct.dataclass
class TrainState:
  step: int
  variables: VariableTreeNode
  opt_state: optax.OptState


@functools.partial(
    jax.jit, static_argnames=["module", "optimizer"]
)
def step(
    module: nn.Module,
    optimizer: optax.GradientTransformation,
    train_state: TrainState,
    x: jax.Array
):
  train_state = dataclasses.replace(train_state, step=(train_state.step + 1))
  def compute_losses(variables: VariableTreeNode, x: jax.Array):
    logits = module.apply(variables, x, method="logits")
    # Use the next token in the example's sequence as the label to evaluate
    # correctness for the loss computation. We lose the final token when
    # computing the loss since we have no meaningful label. C'est la vie.
    labels = jnp.roll(x, shift=-1, axis=-1)

    batch_size, sequence_len, vocab_size = logits.shape
    batch_size, sequence_len = labels.shape
    del batch_size, sequence_len, vocab_size

    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, :-1, :], labels[:, :-1]
    )
    return jnp.mean(losses)

  gradients = jax.grad(compute_losses)(train_state.variables, x)
  parameter_updates, updated_opt_state = optimizer.update(
      gradients, train_state.opt_state, train_state.variables
  )
  train_state = dataclasses.replace(
    train_state,
    variables=optax.apply_updates(train_state.variables, parameter_updates),
    opt_state=updated_opt_state
  )
  return train_state 


if __name__ == "__main__":
  print("Initializing training loop...")

  print("Creating dataset...")

  text = data.load_text("picostories.txt")
  tokenizer = data.CharTokenizer(text)

  # Token sequences from the dataset should be the same length as the context
  # window so that gradients along the context dimension of the positional embedding 
  # table are saturated.
  # 
  # That is, if sequence_len < context_len for training examples, then the rows 
  # between [sequence_len, context_len] in the positional embedding table will 
  # not get any gradients. We want to generate token sequences which are of
  # equal length to the context window to provide gradients to all parts of the
  # positional embedding table.
  dataset = data.create_data_stream(text, tokenizer, batch_size=1, sequence_len=CONTEXT_LEN)

  print("Dataset created, warming up batch generator...")
  warmup_begin = datetime.now()

  sample_batch = next(dataset)

  warmup_end = datetime.now()
  print(f"Got sample batch with shape {sample_batch.shape}, took {str(warmup_end - warmup_begin)}")
  print("Initializing model parameters...")

  gpt = model.GenerativePretrainedTransformer(
      vocab_size=tokenizer.vocab_size,
      embedding_dim=256,
      context_len=CONTEXT_LEN,
      num_heads_per_attn_block=4,
      num_attn_blocks=4
  )
  variables = gpt.init(jax.random.PRNGKey(0), sample_batch)

  print("Model parameters initialized, initializing optimizer...")

  optimizer = optax.adagrad(learning_rate=0.1)
  opt_state = optimizer.init(variables)

  train_state = TrainState(
      step=0,
      variables=variables,
      opt_state=opt_state
  )

  checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
  checkpoint_handler_registry = ocp.DefaultCheckpointHandlerRegistry()
  train_state_save_args = ocp.args.Composite(
      step=ocp.args.ArraySave,
      variables=ocp.args.StandardSave,
      opt_state=ocp.args.StandardSave,
  )
  checkpoint_handler_registry.add(
      item="train_state", args=train_state_save_args
  )
  checkpointer = ocp.CheckpointManager(checkpoint_path, handler_registry=checkpoint_handler_registry)
  print("Auxiliary resources created, enterring training loop!")

  for epoch_num in range(1, NUM_EPOCHS+1):
    print("Began epoch", epoch_num)
    
    epoch_begin = datetime.now()
    for _ in range(STEPS_PER_EPOCH):
      x = next(dataset)
      train_state = step(gpt, optimizer, train_state, x)
    epoch_end = datetime.now()

    print(f"Finished epoch {epoch_num} at step {train_state.step}.")

    epoch_time = (epoch_end - epoch_begin)
    average_step_time = epoch_time / STEPS_PER_EPOCH
    print(f"Epoch took: {epoch_time.seconds} sec")
    print("Average step time: "
          f"{average_step_time.seconds} sec, "
          f"{average_step_time.microseconds} usec")
  
    print("Saving checkpoint.")
    checkpointer.save(
        train_state.step,
        args=ocp.args.Composite(
            step=ocp.args.ArraySave(train_state.step),
            variables=ocp.args.StandardSave(train_state.variables),
            opt_state=ocp.args.StandardSave(train_state.opt_state),
        ),
    )
    print()

  checkpointer.close()
