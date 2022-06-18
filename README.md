# Problems with Cosine as a Measure of Embedding Similarity for High Frequency Words

This repository is meant to help recreate the results from the paper: Problems with Cosine as a Measure of Embedding Similarity for High Frequency Words (https://arxiv.org/abs/2205.05092). The core analysis for the two papers are in ```WiC_analysis.ipynb```,  ```scws_analysis.ipynb```, and ```Section3/wiki_bounding_balls_analysis.ipynb```. In order to reproduce the results, there are five main steps: downloading various data sources (Wikipedia, BookCorpus, WiCs, SCWS), calculating word frequencies from Wikipedia and BookCorpus, querying for keywords from Wikipedia, building contexutalized embeddings, performing analysis on these embeddings. 

In more detail: 

1. **Gather a large corpus of text to create contextualized word embeddings**
Download a large corpus of text which will be used to create contextualized word embeddings. We used the  March 1, 2020 Wikimedia Download which can be found here: https://dumps.wikimedia.org/. We later want to be able to query this corpus for examples of words in context. We scramble this Wikipedia corpus (at the sentence level) so that we can query the first 50 examples found and avoid biases certain Wikipedia pages/topics. 

2. **Calculate word frequencies for Wikipedia and BookCorpus**
We used tools such as wikiextractor and wiki-word-frequencies to help calculate the word frequencies of Wikipedia corpus we have. We then used word frequencies from Complex Word Identification (CWI) Shared Task 2018 project as a way to estimate BookCorpus word frequencies(https://github.com/nathanshartmann/NILC-at-CWI-2018). We combined both frequencies ```calculate_bert_word_frequencies.ipynb``` and the resulting file ```bert_word_frequencies.csv```.

3. **Preprocess WiC, SCWS data**
Download the train and dev datsets for WiC (placed in ```WiC_data```. Preprocess data ```pre_processing_wic.ipynb```. Download the train and dev datsets for SCWS (placed in ```SCWS_data```). Preprocess data using the file: ```scws_embeddings.ipynb```

4. **Create contextualized word embeddings for sentences for the WiC dataset**
This involves embedding the entire sentence and then extract the word embedding associated with the word of interest. We then used Hugging Face's library to create contexutalized word embeddings. See file: ```utils/create_embeddings/creating_word_embeddings.py```. Analyze the results ```WiC_analysis.ipynb``` and ```scws_analysis.ipynb```

5. **Preprocess Wiki data**
30K Random Word Sample
We next selected 30,000 words (across a variety of word frequencies buckets) as our "words of interest". We then query for examples of these words in context in our Wikipedia corpus ```find_words_in_wikipedia.ipynb```. Retain all words with at least 50 examples.

6. **Create contextualized word embeddings for random pairs of sentences from Wikipedia corpus.**
We next selected 30,000 words (across a variety of word frequencies buckets) as our "words of interest". We then queried our Wikipedia corpus for the sentences of these 30,000 words ```find_words_in_wikipedia.ipynb```. 

7. **WiC Word Sample**
We then select all the words **from the WiC task** as "words of interest" and query for examples of these words in context in our Wikipedia corpus. Retain all words with at least 50 examples.

8. **Create contexutalized word embeddings for the Wikiwords**
Repeat step 4 to create contextualized word embeddings for the 30K Random Word Sample and the WiC Word Sample.

9. **Calculate the geometric space occupied by contextualized word embeddings of the same word**
Analyze the geometric space of the random Wikipedia words and the WiC sample words. See files:  ```Section3/creating_word_embedding_bounding_balls.py```, ```Section3/measure_bounding_balls.ipynb```, ```Section3/wiki_bounding_balls_analysis.ipynb```

10. **To how these results compare to other measurements see:**
Measuring the geometric space of contextual embeeding using other metrics such as minimum bounding hyperspheres, norms, and euclidean distance from the centroid. See files: ```Section_3/other_measurements/pre_processing_other_measurements.ipynb``` and ``` Section_3/other_measurements/other_measurements.ipynb```

