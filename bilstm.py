import os 
import numpy as np
import pickle
import re
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter

class BiLSTM:
    def __init__(self, file_pth):
        self.file_pth = file_pth

    def clean_text(self,text):
        text = re.sub(r"[‘’`]", "'", text, flags=re.UNICODE)  
        text = re.sub(r"[“”]", '"', text, flags=re.UNICODE)   
        text = re.sub(r"[â€”–]", "-", text, flags=re.UNICODE)
        text = re.sub(r"[^a-zA-Z0-9\s\.,!?'-]", "", text, flags=re.UNICODE) 
        return text
    
    def make_tokenizer(self):
        with open(self.file_pth, 'r',encoding='utf-8') as f:
            self.text_lines = [self.clean_text(line) for line in f.readlines()]

        self.tokenizer = Tokenizer(oov_token='<oov>',char_level=False,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer.fit_on_texts(self.text_lines)
        
        self.total_words = len(self.tokenizer.word_index) + 1

        print("Total Number of Words in dictionary", self.total_words)
        print("Word ID")
        print(f"<oov>: {self.tokenizer.word_index['<oov>']}")
        print(f"Strong: {self.tokenizer.word_index['strong']}")
        print(f"And: {self.tokenizer.word_index['and']}")

    def get_input_seq(self):
        self.input_seq = []
        for line in self.text_lines:
            token_list = self.tokenizer.texts_to_sequences([line])[0]

            for i in range(1, len(token_list)):
                n_gram_seq = token_list[:i+1]
                self.input_seq.append(n_gram_seq)
       
        print(f"Total input sequences mapped: {len(self.input_seq)}")                

    def padding(self):
        self.max_len = max([len(l) for l in self.input_seq])
        self.input_seq = np.array(pad_sequences(self.input_seq, maxlen=self.max_len, padding='pre'))

    def create_labels(self):
        self.xs, self.labels = self.input_seq[:, :-1], self.input_seq[:, -1]
        self.ys = tf.keras.utils.to_categorical(self.labels, num_classes=self.total_words)
        
    def perplexity(self,y_true, y_pred):
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        perplexity = tf.exp(tf.reduce_mean(cross_entropy))
        return perplexity
        
    def build_model(self, epochs):
        model = Sequential()
        model.add(Embedding(self.total_words, 100, input_length=self.max_len-1))
        model.add(Bidirectional(LSTM(150)))
        model.add(Dense(self.total_words, activation='softmax'))
        adam = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', self.perplexity])
        # history = model.fit(self.xs, self.ys, epochs=epochs, verbose=1)
        history = model.fit(
        self.xs, self.ys, 
        epochs=epochs, 
        validation_split=0.05, 
        verbose=1
        )
        self.plot_training_history(history)
        return model

    def plot_training_history(self, history):
        """Plot training accuracy and loss versus epoch."""
        acc = history.history['accuracy']  
        val_acc = history.history['val_accuracy']  
        epochs = range(1, len(acc) + 1)  
    
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, acc, label='Training Accuracy', marker='o')
        plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
        plt.title('Training and Validation Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(epochs)  
        plt.legend()
        plt.grid()
        plt.show()
    
    def main(self, epochs):
        print("Generating Tokens")
        self.make_tokenizer()
        print("Generating Input Sequence")
        self.report_ngrams()
        print("Generating Input Sequence")
        self.get_input_seq()
        print("Padding")
        self.padding()
        print("Creating Labels")
        self.create_labels()
        print("Building Model")
        self.model = self.build_model(epochs)

    def sample_with_temperature(self,predictions, temperature=0.5):
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-10) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(predictions), p=predictions) 
    
    def _format_as_poem(self, text, num_words):
        words = text.split()[:num_words]
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(current_line) >= 8 or word.endswith(('.', '?', '!')):
                lines.append(' '.join(current_line))
                current_line = []
        if current_line:
            lines.append(' '.join(current_line))
        return '\n'.join(lines)
        
    def report_ngrams(self):
        all_words = []
        for line in self.text_lines:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            words = [word for word, index in self.tokenizer.word_index.items() if index in token_list]
            all_words.extend(words)
        
        bigrams = list(zip(all_words, all_words[1:]))
        trigrams = list(zip(all_words, all_words[1:], all_words[2:]))        
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)        
        print("\nTop 30 Bigrams:")
        for bigram, count in bigram_counts.most_common(30):
            print(f"{bigram}: {count}")
        
        print("\nTop 30 Trigrams:")
        for trigram, count in trigram_counts.most_common(30):
            print(f"{trigram}: {count}")
        
        return bigram_counts, trigram_counts
    import math

        
    def generate(self,seed_text, num_words,temperature=0.5):
        
        for _ in range(num_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_len-1, padding='pre')
            predictions = self.model.predict(token_list, verbose=0)[0]
            predicted = self.sample_with_temperature(predictions, temperature)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break

            seed_text += " " + output_word
            
        gen_poem = self._format_as_poem(seed_text,num_words)  
        print("Generated Poem:")
        print(gen_poem)
if __name__ == '__main__':
    file_pth = '/kaggle/input/updated-raw/updated_raw_text.txt' 
    bilstm = BiLSTM(file_pth)
    bilstm.main(50)
    bilstm.generate('I was not sure', 100,temperature=1)
    pickle.dump(bilstm, open('model_bilstm.pk', 'wb'))
