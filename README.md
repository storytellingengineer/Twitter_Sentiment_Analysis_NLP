# Twitter Sentiment Analysis using NLP

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.


<a class="anchor" id="0.1"></a>
# **Steps**

1.	[Importing the libraries](#1)
2.  [Loading the Dataset](#2)
3.  [Some terminology before going forward](#3)
4.  [Pre-processing the Dataset](#4)
5.  [Using Regular Expression](#5)
6.  [Using SentimentIntensityAnalyzer](#6)
7.  [Calculating scores](#7)


# **1. Importing the libraries** <a class="anchor" id="1"></a>

[Table of Contents](#0.1)

Importing some important libraries for this type of project - Regular expression, NLTK, Spacy

# **2. Loading the Dataset** <a class="anchor" id="2"></a>

[Table of Contents](#0.1)

Using Twitter Dataset downloaded from Kaggle *(Provided in this repository)*

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

# **4. Pre-processing the Dataset** <a class="anchor" id="4"></a>

[Table of Contents](#0.1)

