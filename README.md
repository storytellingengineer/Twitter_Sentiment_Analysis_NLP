# Twitter Sentiment Analysis using NLP

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.


<a class="anchor" id="0.1"></a>
# **Steps**

1.	[Importing the libraries](#1)
2.  [Loading the Dataset](#2)
3.  [Some terminology before going forward](#3)
4.  [Preprocessing the data](#4)
5.  [Exploratory Data Analysis](#5)
6.  [Feature extraction](#6)
7.  [Training the model](#7)


# **1. Importing the libraries** <a class="anchor" id="1"></a>

[Table of Contents](#0.1)

Importing some important libraries like pandas, numpy, NLTK, matplotlib, seaborn...

# **2. Loading the Dataset** <a class="anchor" id="2"></a>

[Table of Contents](#0.1)

Dataset is provided in the repository <a href="https://github.com/AayushSaxena08/Twitter_Sentiment_Analysis_NLP"> Link </a>

# **3. Some terminology before going forward** <a class="anchor" id="3"></a>

[Table of Contents](#0.1)

The Natural Language Toolkit (NLTK) is a platform used for building Python programs that work with human language data for applying in statistical Natural Language Processing (NLP). It contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning.

- **Tokenization**: Tokenization is the process of tokenizing or splitting a string, text into a list of tokens. Before processing a natural language, we want to identify the words that constitute a string of characters. That's why tokenization is a foundational step in NLP. This process is important because the meaning of the text can be interpreted through analysis of the words present in the text.

- **Parsing**: Parsing is used to draw exact meaning or dictionary meaning from the text. It is also called Syntactic analysis or syntax analysis.

- **Classification**: Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups. Different types of classification in NLP - text classification, vector semantic, word embedding, probabilistic language model, sequence labeling, and speech reorganization.
    - **Word Embeddings**: Word embeddings are basically a form of word representation that bridges the human understanding of language to that of a machine. Word embeddings are distributed representations of text in an n-dimensional space. These are essential for solving most NLP problems.
    - **Word2Vec**: The Word2Vec model is used to extract the notion of relatedness across words or products such as semantic relatedness, synonym detection, concept categorization, selectional preferences, and analogy. A Word2Vec model learns meaningful relations and encodes the relatedness into vector similarity.
    - **TF-IDF**: TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of information retrieval and machine learning, that can quantify the importance or relevance of words, phrases, lemmas, etc. in a document amongst a collection of documents (also known as a corpus)
    - **Bag of Words**: Bag of Words (BoW) is a natural language processing (NLP) strategy for converting a text document into numbers that can be used by a computer program. BoW is often implemented as a Python dictionary. Each key in the dictionary is set to a word, and each value is set to the number of times the word appears.
    - **What is the difference between BoW and TF-IDF?** Bag of Words just creates a set of vectors containing the count of word occurrences in the document (reviews), while the TF-IDF model contains information on the more important words and the less important ones as well.
    - **Why Word2Vec is better than bag of words?** The word2vec-based model learns to utilize both textual and visual information, whereas the bag-of-words-based model learns to rely more on textual input.

- **Stemming**: Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma. **Lemmatization** is the grouping together of different forms of the same word. In search queries, lemmatization allows end users to query any version of a base word and get relevant results.
    - Stemming uses the stem of the word, while lemmatization uses the context in which the word is being used.
 
- **Tagging**: POS Tagging in NLTK is a process to mark up the words in text format for a particular part of a speech based on its definition and context. Part of Speech Tags are useful for building parse trees, which are used in building NERs (most named entities are Nouns) and extracting relations between words.

# **4. Preprocessing the data** <a class="anchor" id="4"></a>
[Table of Contents](#0.1)

- removes pattern in the input text
    
        def remove_pattern(input_text,pattern):
          r = re.findall(pattern,input_text)
          for word in r:
            input_text = re.sub(word,  "", input_text)
          return input_text

    
- remove twitter handles (@user)
        
        df["clean_tweet"] = np.vectorize(remove_pattern)(df["tweet"], "@[\w]*")
    
- remove special characters, numbers and punctuations
        
        df["clean_tweet"] = df["clean_tweet"].str.replace(r"[^a-zA-Z0-9]+", ' ')
     
- remove shortcuts
    
        df["clean_tweet"] = df["clean_tweet"].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
    
- Tokenize 
        
        tokenized_tweet = df["clean_tweet"].apply(lambda x:x.split())
      
- Stemming
    
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        tokenized_tweet = tokenized_tweet.apply(lambda sentence:[stemmer.stem(word) for word in sentence])
    
- Combine the words into a sentence
    
        for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = " ".join(tokenized_tweet[i])
        df["clean_tweet"] = tokenized_tweet
    
# **5. Exploratory Data Analysis** <a class="anchor" id="5"></a>
[Table of Contents](#0.1)

- Visualized all the frequently used words and plot a wordcloud 

        all_words = " ".join([sentence for sentence in df["clean_tweet"]])

        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

        #plot the graph
        plt.figure(figsize=(15,8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    ![image](https://user-images.githubusercontent.com/35486320/194721703-cbb6c787-c419-4f3f-85b1-93d79759fb76.png)

- Extracting the hashtags from the tweets

        def hashtag_extract(tweets):
          hashtags = []
          # loop words in a tweet
          for tweet in tweets:
            ht = re.findall(r"#(\w+)", tweet)
            hashtags.append(ht)
          return hashtags
          
        # extract hashtags from non-racist/sexist tweets
        ht_positive = hashtag_extract(df["tweet"][df["label"]==0])

        # extract hashtags from racist/sexist tweets
        ht_negative = hashtag_extract(df["tweet"][df["label"]==1])

- Getting top 10 used words in positive and negative reference

        freq = nltk.FreqDist(ht_positive)
        d = pd.DataFrame({'Hashtag': list(freq.keys()), 'Count':list(freq.values())})
      
        # select top 10 hashtags
        
        d = d.nlargest(columns='Count',n=10)
        plt.figure(figsize=(15,9))
        sns.barplot(data=d,x='Hashtag',y='Count')
        plt.show()
        
     ![image](https://user-images.githubusercontent.com/35486320/194721927-a8289682-795b-460c-87eb-7ef0f0d4aaeb.png)

# **6. Feature extraction** <a class="anchor" id="6"></a>
[Table of Contents](#0.1)

        # Convert into BOW or Word2Vec --- CountVectorizer/TfIdfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
        bow = bow_vectorizer.fit_transform((df['clean_tweet']))

# **7. Training the model** <a class="anchor" id="7"></a>
[Table of Contents](#0.1)

- Used **LogisticRegression** model for binary classification

        from sklearn.linear_model import LogisticRegression
        
- Used **f1 score** to compare models and **accuracy score** to calculate accuracy of the model

     - Getting the accuracy of 94.8% is good enough for LogisticRegression() model

# Thank you for reading this repo. Connect with me on <a href="https://www.linkedin.com/in/aayushsaxena08/">LinkedIn</a>⭐
