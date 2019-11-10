from mpi4py import MPI
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os


# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = open('input.txt', 'r').read()  # should be simple plain text file
data = os.linesep.join([s for s in data.splitlines() if s])
#data = data[:len(data) // 3]

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

# retrieve vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# encode 2 words -> 1 word
sequences = list()

for i in range(2, len(encoded)):
    sequence = encoded[i-2:i+1]
    sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))

# pad sequences
max_length = max([len(seq) for seq in sequences])

# load model from file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')

generate_seq(loaded_model, tokenizer, max_length-1, '', 5)

if rank == 0:
    comm.send('hi its 0', dest=1, tag=11)

if rank == 1:
    d = comm.recv(source=0, tag=11)
    print(d)


