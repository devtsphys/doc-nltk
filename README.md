# Python NLTK Complete Reference Card

## Installation & Setup

```python
# Install NLTK
pip install nltk

# Import and download data
import nltk
nltk.download('punkt')       # Tokenizer
nltk.download('stopwords')   # Stop words
nltk.download('wordnet')     # WordNet corpus
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('vader_lexicon')  # Sentiment analyzer
nltk.download('omw-1.4')     # Open Multilingual Wordnet
```

## Core Modules

### Text Processing & Tokenization

```python
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.tokenize import WhitespaceTokenizer, LineTokenizer

# Basic tokenization
text = "Hello world! This is NLTK. It's amazing."
words = word_tokenize(text)
# ['Hello', 'world', '!', 'This', 'is', 'NLTK', '.', "It's", 'amazing', '.']

sentences = sent_tokenize(text)
# ['Hello world!', 'This is NLTK.', "It's amazing."]

# Custom tokenizers
regex_tokenizer = RegexpTokenizer(r'\w+')
words_only = regex_tokenizer.tokenize(text)
# ['Hello', 'world', 'This', 'is', 'NLTK', 'It', 's', 'amazing']

whitespace_tokenizer = WhitespaceTokenizer()
tokens = whitespace_tokenizer.tokenize("Hello world!\nNew line")
```

### Stop Words

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
text = "This is a sample sentence with stop words"
word_tokens = word_tokenize(text)

filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
# ['sample', 'sentence', 'stop', 'words']

# Available languages
print(stopwords.fileids())
# ['arabic', 'azerbaijani', 'basque', 'bengali', ...]
```

### Stemming

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

# Porter Stemmer (most common)
porter = PorterStemmer()
words = ["running", "runs", "easily", "fairly"]
stemmed = [porter.stem(word) for word in words]
# ['run', 'run', 'easili', 'fairli']

# Snowball Stemmer (supports multiple languages)
snowball = SnowballStemmer('english')
stemmed_snow = [snowball.stem(word) for word in words]
# ['run', 'run', 'easili', 'fairli']

# Lancaster Stemmer (more aggressive)
lancaster = LancasterStemmer()
stemmed_lanc = [lancaster.stem(word) for word in words]
# ['run', 'run', 'easy', 'fair']
```

### Lemmatization

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# Simple lemmatization
words = ["running", "runs", "better", "geese"]
lemmatized = [lemmatizer.lemmatize(word) for word in words]
# ['running', 'run', 'better', 'goose']

# POS-aware lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

words = ["running", "better", "geese"]
pos_lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
# ['run', 'good', 'goose']
```

### Part-of-Speech Tagging

```python
import nltk
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ...]

# Batch tagging
sentences = [
    "I love programming",
    "Python is great"
]
for sentence in sentences:
    tokens = word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    print(tags)
```

### Named Entity Recognition

```python
import nltk
from nltk.tokenize import word_tokenize

text = "Barack Obama was born in Hawaii. He worked in Chicago."
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
named_entities = nltk.ne_chunk(pos_tags)

# Extract named entities
def extract_entities(tree):
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity_name = ' '.join([token for token, pos in subtree.leaves()])
            entity_label = subtree.label()
            entities.append((entity_name, entity_label))
    return entities

entities = extract_entities(named_entities)
# [('Barack Obama', 'PERSON')]
```

### Frequency Distributions

```python
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

text = "the quick brown fox jumps over the lazy dog the fox is quick"
tokens = word_tokenize(text.lower())
fdist = FreqDist(tokens)

# Most common words
print(fdist.most_common(5))
# [('the', 3), ('fox', 2), ('quick', 2), ('brown', 1), ('jumps', 1)]

# Frequency of specific word
print(fdist['the'])  # 3

# Plot frequency distribution
fdist.plot(10)  # Plot top 10 most frequent words
```

### N-grams

```python
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text = "the quick brown fox jumps"
tokens = word_tokenize(text)

# Bigrams (2-grams)
bigrams = list(ngrams(tokens, 2))
# [('the', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumps')]

# Trigrams (3-grams)
trigrams = list(ngrams(tokens, 3))
# [('the', 'quick', 'brown'), ('quick', 'brown', 'fox'), ('brown', 'fox', 'jumps')]

# Character n-grams
char_bigrams = list(ngrams("hello", 2))
# [('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')]
```

### Collocation Detection

```python
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords

text = "The quick brown fox jumps over the lazy dog quickly"
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w not in stop_words]

# Bigram collocations
bigram_finder = BigramCollocationFinder.from_words(filtered_tokens)
bigram_finder.apply_freq_filter(1)  # Minimum frequency
collocations = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 5)
print(collocations)
```

## Text Corpora

### Working with Corpora

```python
from nltk.corpus import brown, reuters, gutenberg, movie_reviews

# Brown Corpus
print(brown.categories())
news_text = brown.words(categories='news')[:100]

# Reuters Corpus
print(reuters.categories()[:10])
reuters_text = reuters.words(categories='grain')[:100]

# Gutenberg Corpus
print(gutenberg.fileids())
emma_text = gutenberg.words('austen-emma.txt')[:100]

# Movie Reviews Corpus
print(movie_reviews.categories())  # ['neg', 'pos']
positive_reviews = movie_reviews.words(categories='pos')[:100]
```

### WordNet

```python
from nltk.corpus import wordnet as wn

# Synsets (synonym sets)
synsets = wn.synsets('car')
print(synsets)  # [Synset('car.n.01'), Synset('car.n.02'), ...]

# Definitions
print(wn.synset('car.n.01').definition())
# 'a motor vehicle with four wheels; usually propelled by an internal combustion engine'

# Examples
print(wn.synset('car.n.01').examples())

# Synonyms
synonyms = []
for syn in wn.synsets('happy'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(set(synonyms))

# Antonyms
antonyms = []
for syn in wn.synsets('happy'):
    for lemma in syn.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
print(set(antonyms))

# Similarity
car = wn.synset('car.n.01')
truck = wn.synset('truck.n.01')
print(car.path_similarity(truck))  # 0.25
```

## Sentiment Analysis

### VADER Sentiment Analyzer

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

texts = [
    "I love this movie!",
    "This movie is terrible.",
    "The movie was okay.",
    "AMAZING!!! Best movie ever!!! :)"
]

for text in texts:
    scores = analyzer.polarity_scores(text)
    print(f"{text}: {scores}")
    # Output: compound score (-1 to 1), neg, neu, pos (0 to 1)
```

## Text Classification

### Naive Bayes Classifier

```python
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk import FreqDist, NaiveBayesClassifier

# Prepare data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Feature extraction
all_words = FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Create feature sets
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

# Train classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy}")

# Show most informative features
classifier.show_most_informative_features(5)

# Classify new text
def classify_text(text):
    tokens = word_tokenize(text.lower())
    features = document_features(tokens)
    return classifier.classify(features)

print(classify_text("This movie was amazing and wonderful!"))
```

## Advanced Text Processing

### Text Similarity

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import math

def text_similarity(text1, text2):
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens1 = [w.lower() for w in word_tokenize(text1) if w.lower() not in stop_words]
    tokens2 = [w.lower() for w in word_tokenize(text2) if w.lower() not in stop_words]
    
    # Calculate Jaccard similarity
    set1, set2 = set(tokens1), set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A fast brown fox leaps over a sleepy dog"
similarity = text_similarity(text1, text2)
print(f"Similarity: {similarity}")
```

### Text Summarization (Simple)

```python
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import heapq

def simple_summarize(text, num_sentences=3):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Tokenize words and remove stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Calculate word frequencies
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1
    
    # Score sentences based on word frequencies
    sentence_scores = defaultdict(float)
    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence.lower())
        for word in sentence_words:
            if word in word_freq:
                sentence_scores[i] += word_freq[word]
        sentence_scores[i] = sentence_scores[i] / len(sentence_words)
    
    # Get top sentences
    top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    top_sentences.sort()
    
    return ' '.join([sentences[i] for i in top_sentences])

# Example usage
long_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. 
It is concerned with the interactions between computers and human language. 
NLP combines computational linguistics with statistical, machine learning, and deep learning models. 
These technologies enable computers to process and analyze large amounts of natural language data. 
Applications of NLP include sentiment analysis, machine translation, and chatbots. 
The field has grown rapidly with advances in machine learning and big data.
"""

summary = simple_summarize(long_text, 2)
print(summary)
```

### Regular Expressions with NLTK

```python
import re
from nltk.tokenize import RegexpTokenizer

# Custom tokenizers using regex
email_tokenizer = RegexpTokenizer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
url_tokenizer = RegexpTokenizer(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

text = "Contact me at john@example.com or visit https://example.com"
emails = email_tokenizer.tokenize(text)
urls = url_tokenizer.tokenize(text)

print(f"Emails: {emails}")
print(f"URLs: {urls}")

# Phone number extraction
phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
text_with_phones = "Call me at 555-123-4567 or 555.987.6543"
phones = re.findall(phone_pattern, text_with_phones)
print(f"Phone numbers: {phones}")
```

## Parsing and Grammar

### Chunking

```python
import nltk
from nltk.chunk import RegexpParser

# Define chunk grammar
grammar = r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
        {<NNP>+}                # chunk sequences of proper nouns
"""

chunk_parser = RegexpParser(grammar)

sentence = "The little yellow dog barked at the cat"
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
chunked = chunk_parser.parse(pos_tags)

print(chunked)
chunked.draw()  # Visual representation
```

## Performance and Optimization

### Text Preprocessing Pipeline

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens

# Usage
preprocessor = TextPreprocessor()
text = "The quick brown foxes are running through the forest!"
processed = preprocessor.preprocess(text)
print(processed)  # ['quick', 'brown', 'fox', 'running', 'forest']
```

## Common Utility Functions

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

def word_count(text):
    """Count total words in text"""
    tokens = word_tokenize(text)
    return len(tokens)

def unique_words(text):
    """Count unique words in text"""
    tokens = word_tokenize(text.lower())
    return len(set(tokens))

def average_word_length(text):
    """Calculate average word length"""
    tokens = word_tokenize(text)
    word_tokens = [token for token in tokens if token.isalpha()]
    if not word_tokens:
        return 0
    return sum(len(word) for word in word_tokens) / len(word_tokens)

def lexical_diversity(text):
    """Calculate lexical diversity (unique words / total words)"""
    tokens = word_tokenize(text.lower())
    word_tokens = [token for token in tokens if token.isalpha()]
    if not word_tokens:
        return 0
    return len(set(word_tokens)) / len(word_tokens)

def most_common_words(text, n=10, remove_stopwords=True):
    """Find most common words"""
    tokens = word_tokenize(text.lower())
    word_tokens = [token for token in tokens if token.isalpha()]
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        word_tokens = [token for token in word_tokens if token not in stop_words]
    
    return Counter(word_tokens).most_common(n)

# Example usage
text = "The quick brown fox jumps over the lazy dog. The dog was really lazy."
print(f"Word count: {word_count(text)}")
print(f"Unique words: {unique_words(text)}")
print(f"Average word length: {average_word_length(text):.2f}")
print(f"Lexical diversity: {lexical_diversity(text):.2f}")
print(f"Most common words: {most_common_words(text, 5)}")
```

## Error Handling and Best Practices

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def safe_tokenize(text):
    """Safely tokenize text with error handling"""
    try:
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            return []
        return word_tokenize(text)
    except Exception as e:
        print(f"Tokenization error: {e}")
        return text.split()

def safe_pos_tag(tokens):
    """Safely perform POS tagging"""
    try:
        return nltk.pos_tag(tokens)
    except Exception as e:
        print(f"POS tagging error: {e}")
        return [(token, 'NN') for token in tokens]  # Default to noun

# Check if required data is downloaded
def ensure_nltk_data():
    """Ensure required NLTK data is available"""
    required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            print(f"Downloading {data}...")
            nltk.download(data)

# Usage
ensure_nltk_data()
text = "This is a test sentence."
tokens = safe_tokenize(text)
pos_tags = safe_pos_tag(tokens)
```

## Quick Reference

### Common Import Statements

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
```

### Essential Downloads

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('movie_reviews')
```

### POS Tags Reference

- **NN**: Noun (singular)
- **NNS**: Noun (plural)
- **VB**: Verb (base form)
- **VBD**: Verb (past tense)
- **VBG**: Verb (gerund/present participle)
- **JJ**: Adjective
- **RB**: Adverb
- **DT**: Determiner
- **PRP**: Personal pronoun
- **IN**: Preposition

This reference card covers the most important NLTK functions and techniques for natural language processing tasks. Each section includes practical examples that you can run directly in your Python environment.