
import numpy as np
import pandas as pd
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ast import literal_eval
import pickle
from sklearn.model_selection import train_test_split
import argparse
import spacy

class preprocessing():
    def __init__(self):
        self.maxlen = 110
        self.max_words = 36000
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.model_spacy = spacy.load("en_core_web_sm")

    def load_data(self, file_id):
        """Load data with error handling."""
        try:
            # Memuat dataset dengan pandas read_csv dari hasil download file google drive https://drive.google.com/file/d/1OoaUzSoFI-ZwHMQ55vr3MBpuNDVtJ_CX/view?usp=sharing
            gdown.download(f"https://drive.google.com/uc?id={file_id}", "ner.csv", quiet=False)
            df = pd.read_csv("ner.csv", encoding = "unicode_escape", on_bad_lines='skip', usecols=['sentence_idx', 'word', 'tag'])
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("Error: File not found.")
        except pd.errors.ParserError:
            print("Error: Failed to parse CSV file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

    def handling_missing_value(self, df):
        """Handling Missing Value for every columns"""
        try:
            df = df[df['sentence_idx'] != 'prev-lemma'].dropna(subset=['sentence_idx']).reset_index(drop=True)
            df = df.dropna(subset=["sentence_idx", "word", "tag"])
            print(f"Data handling missing value successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return df

    def reformat_dataframe(self, df):
        """Reformat the dataframe"""
        if df is None:
            print("No dataframe to preprocess.")
            return None
        try:
            # Membuat bentuk kalimat dari data 
            words = pd.DataFrame(df.groupby('sentence_idx')['word'].apply(list))
            tags = df.groupby('sentence_idx')['tag'].apply(list)
            words['tag'] = tags
            words['sentence'] = words['word'].apply(lambda x: ' '.join(x))
            words['tag_combine'] = words['tag'].apply(lambda x: ' '.join(x))
            words = words.reset_index()
            print("Reformat dataset completed successfully.")
            return words
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None      
    
    def split_dataset_ML(self, words):
        try:
            train_data, test_data, train_label, test_label = train_test_split(words['sentence'], words['tag_combine'], test_size=0.20, random_state=42)
            train_data = train_data.tolist()
            test_data = test_data.tolist()
            train_label = train_label.tolist()
            test_label = test_label.tolist()
            print("Split dataset Machine Learning completed successfully.")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None 
        return train_data, test_data, train_label, test_label

    # Function untuk mendapatkan nilai fitur untuk satu kata
    def getFeaturesForOneWord(self, sentence, pos, pos_tags):
        word = sentence[pos]

        features = [
            'word.lower=' + word.lower(), # serves as word id
            'word[-3:]=' + word[-3:],     # last three characters
            'word[-2:]=' + word[-2:],     # last two characters
            'word.isupper=%s' % word.isupper(),  # is the word in all uppercase
            'word.isdigit=%s' % word.isdigit(),  # is the word a number
            'word.startsWithCapital=%s' % word[0].isupper(), # is the word starting with a capital letter
            'word.pos=' + pos_tags[pos]
        ]

        #Use the previous word also while defining features
        if(pos > 0):
            prev_word = sentence[pos-1]
            features.extend([
            'prev_word.lower=' + prev_word.lower(), 
            'prev_word.isupper=%s' % prev_word.isupper(),
            'prev_word.isdigit=%s' % prev_word.isdigit(),
            'prev_word.startsWithCapital=%s' % prev_word[0].isupper(),
            'prev_word.pos=' + pos_tags[pos-1]
        ])
        # Mark the begining and the end words of a sentence correctly in the form of features.
        else:
            features.append('BEG') # feature to track begin of sentence 

        if(pos == len(sentence)-1):
            features.append('END') # feature to track end of sentence
        return features

    def getFeaturesForOneSentence(self, sentence):
        # We need to get the pos_tags to be passed to the function
        processed_sent = self.model_spacy(sentence)
        postags = []
        
        for each_token in processed_sent:
            postags.append(each_token.pos_)
        
        sentence_list = sentence.split()
        return [self.getFeaturesForOneWord(sentence_list, pos, postags) for pos in range(len(sentence_list))]
    
    def getLabelsInListForOneSentence(self, labels):
        return labels.split()
    
    
    def remove_uneven_len(self, train_x, train_y):
        for i in range(len(train_x)):
            if (len(train_x)-1 == i):
                return False, train_x, train_y
            elif(len(train_x[i]) != len(train_y[i])):
                del train_x[i]
                del train_y[i]
                return True, train_x, train_y
    
    def normalize_ML(self, train_data, test_data, train_label, test_label):
        try:
            ret = True
            X_train = [self.getFeaturesForOneSentence(sentence) for sentence in train_data]
            X_test = [self.getFeaturesForOneSentence(sentence) for sentence in test_data]

            Y_train = [self.getLabelsInListForOneSentence(labels) for labels in train_label]
            Y_test = [self.getLabelsInListForOneSentence(labels) for labels in test_label]

            while(ret):
                ret, X_train, Y_train = self.remove_uneven_len(Y_train, Y_train)
            print("Normalize Data Machine Learning completed successfully.")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None
        return X_train, X_test, Y_train, Y_test
    
    def preprocess_data(self, words):
        X = list(words['sentence'])
        Y = list(words['tag'])

        Y_ready = []
        for sen_tags in Y:
            Y_ready.append(literal_eval(str(sen_tags)))
        try:
            self.tokenizer.fit_on_texts(X)
            sequences = self.tokenizer.texts_to_sequences(X)
            word_index = self.tokenizer.word_index
            print("Data Tokenizer completed successfully.")
        except Exception as e:
            print(f"Error during process data tokenizer: {e}")
            return None
        # Save tokenizer for prediction
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)
        
        word2id = word_index
        id2word = {}
        try:
            for key, value in word2id.items():
                id2word[value] = key
            X_preprocessed = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
            print("Padding Squence completed successfully.")
        except Exception as e:
            print(f"Error during process data padding squence: {e}")
            return None
        
        return Y_ready, X_preprocessed
    
    def label_process(self, tags):
        try:
            tags2id = {}
            for i, tag in enumerate(tags):
                tags2id[tag] = i

            id2tag = {}
            for key, value in tags2id.items():
                id2tag[value] = key
            print("Label Procsessing completed successfully.")
            return tags2id, id2tag
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None
    
    def preprocess_tags(self, tags2id, Y_ready):
        Y_preprocessed = []
        maxlen = 110
        i = 0
        for y in Y_ready:
            j = 0
            Y_place_holder = []
            
            for tag in y:
                if(j >= 110):
                    pass
                else:
                    Y_place_holder.append(tags2id[tag])
                j +=1
            
            len_new_tag_list = len(Y_place_holder)
            num_O_to_add = maxlen - len_new_tag_list
            
            padded_tags = Y_place_holder + ([tags2id['O']] * num_O_to_add)
            Y_preprocessed.append(padded_tags)
        return Y_preprocessed
    
    def split_dataset_DL(self, X_preprocessed, Y_preprocessed, file_path):
        try:
            indices = np.arange(len(Y_preprocessed))
            np.random.seed(seed=555)
            np.random.shuffle(indices)
            X_preprocessed = X_preprocessed[indices]
            Y_preprocessed = Y_preprocessed[indices]
            
            X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y_preprocessed, test_size=0.2, random_state=42)

            # Simpan semuanya sebagai dictionary
            data = {
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test
            } 
            with open("automate_{}_DL.pkl".format(file_path), "wb") as f:
                pickle.dump(data, f)

            print("Split Dataset completed successfully.")
        except Exception as e:
            print(f"Error during split dataset: {e}")
            return False
        return True
    
    def prepare_ML(self, words, file_path):
        train_data, test_data, train_label, test_label = self.split_dataset_ML(words)
        X_train, X_test, Y_train, Y_test = self.normalize_ML(train_data, test_data, train_label, test_label)
        data_ml = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": Y_test
        }
        with open("automate_{}_ML.pkl".format(file_path), "wb") as f:
            pickle.dump(data_ml, f)
        return True

    def prepare_DL(self, words, tags, file_path):
        Y_ready, X_preprocessed = self.preprocess_data(words)
        tags2id, id2tag = self.label_process(tags)
        Y_preprocessed = self.preprocess_tags(tags2id, Y_ready)
        X_preprocessed = np.asarray(X_preprocessed)
        Y_preprocessed = np.asarray(Y_preprocessed)
        status = self.split_dataset_DL(X_preprocessed, Y_preprocessed, file_path)
        return status
    
    def main(self, drive_id, file_path):
        df = self.load_data(drive_id)
        df = self.handling_missing_value(df)
        words = self.reformat_dataframe(df)
        tags = df["tag"].unique()
        status_ML = self.prepare_ML(words, file_path)
        status_DL = self.prepare_DL(words, tags, file_path)
        return status_ML, status_DL

# Example usage (you can comment out this section if using as a module)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_result", type=str, default="ner_dataset")
    args = parser.parse_args()
    file_path = args.name_result
    prep = preprocessing()
    drive_id = "1OoaUzSoFI-ZwHMQ55vr3MBpuNDVtJ_CX"  # Change this to your CSV file path
    status_ML, status_DL = prep.main(drive_id, file_path)
