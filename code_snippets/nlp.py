# Import Libraries
import subprocess
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel

# Download and unzip wordnet if not present (Required for importing wordnet)
try:
    nltk.data.find('wordnet.zip')
except:
    nltk.download('wordnet')
    nltk_data_path = nltk.data.find('corpora/wordnet.zip')
    nltk_data_path = str(nltk_data_path)
    # print(str(nltk_data_path),nltk_data_path)
    command = f"unzip {nltk_data_path} -d {nltk_data_path.rsplit('/', 1)[0]}"
    # command = "unzip /kaggle/working/corpora/wordnet.zip -d /kaggle/working/corpora"
    subprocess.run(command.split())
    nltk.data.path.append(nltk_data_path.rsplit('/', 1)[0])
    # nltk.data.path.append('/kaggle/working/')

# Download and unzip stopwords if not present (Required for importing stopwords)
print(nltk.data.find('corpora'))
if 'corpora/stopwords' in nltk.data.find('corpora'):
    print("Stopwords are already downloaded!")
else:
    print("Stopwords are not downloaded. Downloading now...")
    nltk.download('stopwords')

# import if wordnet and stopwords is present
from nltk.corpus import wordnet, stopwords

class text_utils:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = stopwords.words('english')

    def clean(self, text): 
        text=text.lower()
        obj=re.compile(r"<.*?>") #removing html tags
        text=obj.sub(r" ",text)
        obj=re.compile(r"https://\S+|http://\S+") #removing url
        text=obj.sub(r" ",text)
        obj=re.compile(r"[^\w\s]") #removing punctuations
        text=obj.sub(r" ",text)
        obj=re.compile(r"\d{1,}") #removing digits
        text=obj.sub(r" ",text)
        obj=re.compile(r"_+") #removing underscore
        text=obj.sub(r" ",text)
        obj=re.compile(r"\s\w\s") #removing single character
        text=obj.sub(r" ",text)
        obj=re.compile(r"\s{2,}") #removing multiple spaces
        text=obj.sub(r" ",text)  
        return text

    def pos_tagger(self,nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def preprocess_sent(self,sent, lemmatize=True,stem=True, string_op=True):
        sent = self.clean(sent)
        sent_list = self.tokenizer.tokenize(sent)
        if lemmatize:
            sent_list = nltk.pos_tag(sent_list)
            sent_list = [i for i in sent_list if i[0] not in self.stopwords]
            sent_list = list(map(lambda t: (t[0],self.pos_tagger(t[1])),sent_list))
            sent_list = [i[0] if i[1] is None else self.lemmatizer.lemmatize(i[0],i[1]) 
                                                    for i in sent_list]
        else:
            sent_list = [i for i in sent_list if i not in self.stopwords]
        if stem:
            sent_list = [self.stemmer.stem(i) for i in sent_list]
        if string_op:
            return " ".join(sent_list)
        else:
            return sent_list

class word_embeddings:
    def __init__(self):
        pass

    def gensim_model(self,data,col_name):
        model = Word2Vec(vector_size=100, window=5)
        model.build_vocab(data[col_name])
        model.train(data[col_name], total_examples=model.corpus_count, epochs=5)
        return model

    def gensim_embeddings(self, word_list, gensim_model):
        num_vector = np.array([0]*gensim_model.vector_size)
        count = 0
        for word in word_list:
            if word in gensim_model.wv:
                num_vector = num_vector + gensim_model.wv[word]
                count +=1
        num_vector = num_vector/4
        return num_vector

    def pytorch_embeddings(self, text, model_name='bert-base-uncased'):
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        embeddings = outputs.last_hidden_state
        sentence_embedding = embeddings.mean(dim=1).squeeze()
    
        return sentence_embedding

if __name__ == "__main__":
    text_process_obj = text_utils()
    print("Code Run Complete")