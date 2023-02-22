from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
import tensorflow as tf
from tensorflow.keras.layers import Input, LayerNormalization, Dense, Dropout, MultiHeadAttention, Add
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW


def build_model(input_shape, af='elu'):
    input_layer = Input(shape=input_shape, name='inputs')
    outputs = TransformerEncoder(n_layers=8, d_model=input_shape[-1], n_head=12, d_ff=4096)(input_layer)
    # outputs = Reshape((2, 12), name='outputs')(x)
    return Model(inputs=[input_layer], outputs=[outputs])


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(num_heads=n_head, key_dim=d_model)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(d_ff, activation='relu')
        self.dense2 = Dense(d_model)
        self.dropout2 = Dropout(dropout)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False, mask=None):
        # Multi-head attention
        attn_output = self.multi_head_attention(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        # Feedforward network
        ffn_output = self.dense2(self.dropout2(self.dense1(out1), training=training))
        out2 = self.norm2(out1 + ffn_output)

        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, n_head, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.encoder_layers = [TransformerEncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)]

    def call(self, x, training=False, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, training=training, mask=mask)
        return x


def main():

    # Set the random seed for reproducibility
    tf.random.set_seed(42)

    # Define the hyperparameters
    vocab_size = 10000
    max_seq_len = 128
    d_model = 768
    nhead = 12
    num_layers = 12
    dropout = 0.1
    batch_size = 64
    num_epochs = 10
    learning_rate = 5e-4

    # Load the data as a list of strings
    with open("plrabn12.txt", "r") as f:
        data = f.readlines()

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Preprocess the data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(train_data)
    train_dataset = tokenizer.texts_to_sequences(train_data)
    train_dataset = tf.keras.preprocessing.sequence.pad_sequences(train_dataset, maxlen=max_seq_len, padding="post", truncating="post")
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.map(lambda x: (x[:-1], x[1:]))
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tokenizer.texts_to_sequences(test_data)
    test_dataset = tf.keras.preprocessing.sequence.pad_sequences(test_dataset, maxlen=max_seq_len, padding="post", truncating="post")
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
    test_dataset = test_dataset.map(lambda x: (x[:-1], x[1:]))
    test_dataset = test_dataset.batch(batch_size)

    # Build the model
    model = TransformerEncoder(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)

    # Define the loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Define the checkpoint callback to save the model weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("model_weights.h5", save_weights_only=True)

    model.fit(train_dataset, epochs=num_epochs, callbacks=[checkpoint_callback])

    # Train the model and evaluate on the test set after each epoch
    # for epoch in range(num_epochs):
    #     # Train the model on the training set
    #     model.fit(train_dataset, epochs=1, callbacks=[checkpoint_callback])
        
    #     # Evaluate the model on the test set
    #     loss, accuracy = model.evaluate(test_dataset)
    #     print("Epoch {:d} - Test Loss: {:.4f} - Test Accuracy: {:.4f}".format(epoch+1, loss, accuracy))


# class TextDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, index):
#         return torch.tensor(self.data[index])

#     def __len__(self):
#         return len(self.data)
    
# def train():
#     text_file_path = "path/to/text/file.txt"
#     with open(text_file_path, "r", encoding="utf-8") as f:
#         text = f.read()

#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     tokenized_text = tokenizer.encode(text)
    
#     dataset = TextDataset(tokenized_text)

#     batch_size = 16
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     model_name = "gpt2"
#     model = AutoModelForCausalLM.from_pretrained(model_name)
    
#     optimizer = AdamW(model.parameters(), lr=2e-5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

#     # Train the model
#     n_epochs = 10
#     for epoch in range(n_epochs):
#         total_loss = 0.0
#         for batch in data_loader:
#             batch = batch.to(model.device)

#             # Shift the input to the right by one token (this is necessary for the causal language modeling task)
#             input_ids = batch[:, :-1]
#             labels = batch[:, 1:]

#             # Compute the loss
#             outputs = model(input_ids=input_ids, labels=labels)
#             loss = outputs.loss

#             # Backpropagate and update the model parameters
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         # Update the learning rate
#         scheduler.step()
        
#         print("Epoch {}: average loss = {}".format(epoch+1, total_loss/len(data_loader)))


# def test():
#     test_text_file_path = "path/to/test/text/file.txt"
#     with open(test_text_file_path, "r", encoding="utf-8") as f:
#         test_text = f.read()
#     test_tokenized_text = tokenizer.encode(test_text)
#     test_dataset = TextDataset(test_tokenized_text)
#     test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     model.eval()
#     total_log_likelihood = 0.0
#     n_words = 0
#     with torch.no_grad():
#         for batch in test_data_loader:
#             batch = batch.to(model.device)

#             input_ids = batch[:, :-1]
#             labels = batch[:, 1:]

#             outputs = model(input_ids=input_ids, labels=labels)
#             log_likelihoods = outputs.logits.view(-1, tokenizer.vocab_size).gather(1, labels.view(-1, 1))
#             total_log_likelihood += log_likelihoods.sum().item()
#             n_words += labels.numel()

#     log_likelihood = total_log_likelihood / n_words
#     perplexity = torch.exp(-log_likelihood).item()

#     print("Perplexity on test set = {}".format(perplexity))
    
# def main():
#     # todo

if __name__ == "__main__":
    main()