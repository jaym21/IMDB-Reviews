import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import io

imdb, info = tfds.load("imdb_reviews", with_info = True, as_supervised = True)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for s,l in train_data:
    training_sentences.append(s.numpy().decode('utf-8'))
    training_labels.append(l.numpy())

for s,l in test_data:
    testing_sentences.append(s.numpy().decode('utf-8'))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final  = np.array(testing_labels)

#initializing for easy changes
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>" #for out of vocabulary words

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
#tokenizing the training sentences
tokenizer.fit_on_texts(training_sentences) 
#giving unique index to each word in training data
word_index = tokenizer.word_index 
#making sentences by using word indices 
sequences = tokenizer.texts_to_sequences(training_sentences) 
#padding the sequence and making each sequence of word indices equal in size by adding zeroes at the end of the sequence
training_padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

#making sequences and then padding them on testing sentences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(training_padded[3]))
print(training_sentences[3])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(training_padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

#getting the weights of the easch layer in model
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num] 
    out_m.write(word + "\n") #writing actual wod associated 
    out_v.write("\t".join([str(x) for x in embeddings]) + "\n") #writing all values for 16 dimensions in vector form
out_m.close()
out_v.close()

#this used to download the words and vector files created out of the our model(this piece of code works only in jupyter notebook)
# try:
#   from google.colab import files
# except ImportError:
#   pass
# else:
#   files.download('vecs.tsv')
#   files.download('meta.tsv')