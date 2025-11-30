import jax.numpy as jnp
import numpy as np

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, string):
        """Converts a string to a list of integers."""
        return [self.stoi[c] for c in string]

    def decode(self, int_list):
        """Converts a list of integers back to a string."""
        return ''.join([self.itos[i] for i in int_list])

def load_text(filepath):
    """Loads text data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_data_stream(text, tokenizer, batch_size, sequence_len):
    """Creates a generator that yields batches of data."""
    # Encode the entire dataset
    data = np.array(tokenizer.encode(text))
    
    # Calculate the number of full batches we can make
    num_chars = len(data)
    num_batches = num_chars // (batch_size * sequence_len)
    
    if num_batches == 0:
        raise ValueError("Not enough data to create a single batch. Try smaller batch_size or sequence_len.")

    # Trim data to fit full batches
    data = data[:num_batches * batch_size * sequence_len]
    
    # Reshape into batches
    data = data.reshape((batch_size, -1))
    
    while True:
        # Split the data into chunks of sequence_len
        for i in range(0, data.shape[1], sequence_len):
            chunk = data[:, i:i + sequence_len]
            if chunk.shape[1] == sequence_len:
                yield jnp.asarray(chunk)

# Example of how to use this in your trainer.py:
if __name__ == '__main__':
    # 1. Load the text data
    text = load_text('tinystories.txt')
    
    # 2. Create the tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # 3. Create the data stream
    batch_size = 32
    sequence_len = 128
    data_stream = create_data_stream(text, tokenizer, batch_size, sequence_len)
    
    # 4. Fetch a batch
    sample_batch = next(data_stream)
    print(f"Sample batch shape: {sample_batch.shape}")
    
    # 5. Decode a sample from the batch to verify
    decoded_sample = tokenizer.decode(sample_batch[0].tolist())
    print("\n--- Decoded Sample ---")
    print(decoded_sample)
    print("----------------------")
