import jax
import jax.numpy as jnp

import flax.linen as nn


class CausalSelfAttention(nn.Module):
  """Home-cooked implementation of a causal, single-headed, self-attention layer.
  
  Mostly taken from karpathy/mingpt and Attention is All You Need.

  See 3Blue1Brown's explanation of attention in transformers:
  https://youtu.be/eMlx5fFNoYc.
  """

  key_size: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    keys: jax.Array = nn.Dense(self.key_size)(x)
    queries: jax.Array = nn.Dense(self.key_size)(x)

    # Represents how important other elements in the sequence are to affecting the
    # meaning of `x`. That is, what other elements of the sequence "attend to `x`"
    attention_pattern = nn.softmax(self.causal_mask(
        (queries @ keys.transpose(0, 2, 1)) / jnp.sqrt(self.key_size)
    ))

    # Note that the size of the value vector doesn't need to be the same as the
    # key vector. This value is d_model in Attention is All You Need. In
    # practice, it's usually the case that d_model == d_k.
    values: jax.Array = nn.Dense(self.key_size)(x)

    # An individual row in this tensor represents the semantic effect other tokens
    # have on `x`. That is, it represents how much other elements in the sequence augment
    # the semantics of `x` in its embedding space.
    semantic_perturbations = attention_pattern @ values
    return semantic_perturbations

  def causal_mask(self, tensor: jax.Array) -> jax.Array:
    """Take only the upper-triangle of `tensor` by masking its lower triangle with -inf.

    When `tensor` is an attention pattern, its lower-triange represents 
    attention scores for "future" tokens. A motivation for omitting these
    tokens in softmax might be that we don't want the model to learn the
    "answers" provided by the training data.
    """
    sequence_len = tensor.shape[2]
    mask = jnp.triu(jnp.ones((sequence_len, sequence_len)), k=-1)
    return jnp.where(mask == 0, -jnp.inf, tensor)


class MultiHeadAttention(nn.Module):

  embedding_dim: int
  num_heads: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass on a multi-head attention layer.

    Mechanically, multi-head attention works by splitting a single embedding
    space across multiple heads. Each head attends to its own part of the
    embedding space independently. A final projection layer is used to allow
    the model to learn the relationship between the different heads.

    Semantically, each head can be thought of as "attending to" a particular
    aspect of the embedding space. For example, one head could be learning
    grammatical relationships between tokens, while another can be learning
    semantic meaning. In practice, interpreting the actual semantic behavior 
    of heads is hard.
    """
    # Concat across the `value` axis to provide attention information from all 
    # heads to each element in the sequence for each example in the batch.
    semantic_perturbations = jnp.concat([
        CausalSelfAttention(key_size=(self.embedding_dim // self.num_heads))(x)
        for _ in range(self.num_heads)
    ], axis=2)
    semantic_perturbations = nn.Dense(self.embedding_dim)(semantic_perturbations)
    return semantic_perturbations


class GPTAttentionBlock(nn.Module):

  embedding_dim: int
  num_heads: int

  def setup(self):
    self.feed_forward_network = nn.Sequential([
        nn.LayerNorm(),
        nn.Dense(4 * self.embedding_dim),
        nn.gelu,
        nn.Dense(self.embedding_dim),
    ])

  @nn.compact
  def __call__(self, embedding: jax.Array) -> jax.Array:
    normalized_embedding = nn.LayerNorm()(embedding)
    semantic_perturbations = MultiHeadAttention(
        embedding_dim=self.embedding_dim,
        num_heads=self.num_heads
    )(normalized_embedding)
    augmented_embedding = embedding + semantic_perturbations
    return augmented_embedding + self.feed_forward_network(augmented_embedding)


class GenerativePretrainedTransformer(nn.Module):

  vocab_size: int
  embedding_dim: int
  context_len: int
  num_attn_blocks: int
  num_heads_per_attn_block: int

  def setup(self):
    self.token_embedding = nn.Embed(self.vocab_size, self.embedding_dim)
    self.positional_embedding = nn.Embed(self.context_len, self.embedding_dim)

    self.attention_blocks = [GPTAttentionBlock(self.embedding_dim, self.num_heads_per_attn_block)
                             for _ in range(self.num_attn_blocks)]

  @nn.compact
  def __call__(self, token_id_sequences: jax.Array) -> jax.Array:
    batch_size, sequence_len = token_id_sequences.shape

    embedded_sequences = (self.token_embedding(token_id_sequences) +
                          self.positional_embedding(jnp.arange(sequence_len)))
    next_token_embeddings = nn.Sequential([
        nn.LayerNorm(),
        *self.attention_blocks,
        nn.LayerNorm()
    ])(embedded_sequences)
    next_token_embeddings = jnp.reshape(
        next_token_embeddings, (batch_size * sequence_len, self.embedding_dim)
    )

    flat_logits = next_token_embeddings @ self.token_embedding.embedding.T
    logits = jnp.reshape(flat_logits, (batch_size, sequence_len, self.vocab_size))
    return logits
