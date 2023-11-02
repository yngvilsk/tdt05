import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
#from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Dense

# Load datasets
df_radical = pd.read_csv('radical_data.csv', nrows=10)
df_non_radical = pd.read_csv('neutral_data.csv', encoding='ISO-8859-1')

# Filter negative tweets and sample 30
negative_tweets = df_non_radical[df_non_radical.iloc[:, 0] == 0].sample(5)

positive_tweets = df_non_radical[df_non_radical.iloc[:, 0] == 4].sample(5)


# Concatenate the sampled tweets to get a total of 90 tweets
df_neutral = pd.concat([negative_tweets, positive_tweets])


# Preprocess data
def preprocess(tweet):
     # A number of the tweets start with ENGLISH TRANSLATIONS: so i will remove it 
    tweet = re.sub(r'ENGLISH TRANSLATION:','',tweet)
    #I will also strip the tweets of non-alphabetic characters except #
    tweet = re.sub(r'[^A-Za-z# ]','',tweet)
    
    words = tweet.strip().split()
  
    # Convert hashtags by just removing the '#' and keep the word
    words = [word.replace('#', '').lower() for word in words]
    
    # remove stopwords and stem words using porter stemmer
    p_stem = PorterStemmer()
    words = [p_stem.stem(word.lower()) for word in words if word not in stopwords.words('english')]
    
    #words = list(set(words))
    return words


# Applying the function to the 'tweets' column of both datasets
radical_tweets = df_radical['tweets'].apply(preprocess)
neutral_tweets = df_neutral.iloc[:, -1].apply(preprocess)  # Assuming the last column contains the tweets

# Combine the preprocessed tweets from both datasets
all_tweets = pd.concat([radical_tweets, neutral_tweets])


# Combine and split data
# X = ...
# y = ...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Word2Vec model
model_w2v = Word2Vec(sentences=all_tweets.tolist(), vector_size=50, window=3, min_count=2, workers=4)
model_w2v.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

# Convert tweets to vectors
#def tweet_to_vector(tweet, model):
 #   return np.mean([model.wv[word] for word in tweet if word in model.wv.index_to_key], axis=0)

#X_train_vec = np.array([tweet_to_vector(tweet, model_w2v) for tweet in X_train])
#X_test_vec = np.array([tweet_to_vector(tweet, model_w2v) for tweet in X_test])

# Neural network
#model_nn = Sequential()
#model_nn.add(Dense(128, activation='relu', input_dim=100))
#model_nn.add(Dense(64, activation='relu'))
#model_nn.add(Dense(1, activation='sigmoid'))

#model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model_nn.fit(X_train_vec, y_train, epochs=10, batch_size=32, validation_data=(X_test_vec, y_test))
