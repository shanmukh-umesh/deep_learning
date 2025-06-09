import pandas as pd
import glob
from nltk.corpus import stopwords
import nltk
import re
from bs4 import BeautifulSoup
import urllib.request
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm 
import pyphen
import numpy as np

"""
url imput
"""
url_df = pd.read_excel('./Input.xlsx')

"""
read postive words and negative words defined in masterdirectory
"""
are_positive_df = pd.read_csv('./MasterDirectory/positive-words.txt',header = None)
are_negative_df = pd.read_csv('./MasterDirectory/negative-words.txt',encoding='latin-1',header = None)
predefined_positive_words = list(are_positive_df[0])
predefined_negative_words = list(are_negative_df[0])


"""
to read all text files inside stopwords folder 
"""
stopwords_filepath_list = glob.glob('./Stopwords/*.txt')
stopwords_file_list = []
for path in stopwords_filepath_list:
    file = path.split('\\')[1]
    stopwords_file_list.append(file)
# read stopwords from all text files inside stopwords directory
my_stopwords = set()
for i in stopwords_filepath_list:
    try:
        temp_df = pd.read_csv(i,sep='|',header=None)
    except UnicodeDecodeError:
        temp_df = pd.read_csv(i,sep='|',encoding='latin-1',header=None)
    except Exception as e:
        print("Error while reading Stopwords CSV files data",e)
    my_stopwords = my_stopwords.union(set(temp_df[0]))
default_stopwords = stopwords.words('english')
custom_stopwords = my_stopwords.union(default_stopwords)
# support function 
def support_webdata(url):
    with urllib.request.urlopen(url) as response:
        if response.status == 200:
            content_type = response.getheader('Content-Type')
            if 'text/html' in content_type:
                html_content = response.read().decode('utf-8')
                # html5lib
                # lxml
                soup = BeautifulSoup(html_content,'lxml')
                # print(soap)
            else:
                # print(f"Unexpected content type: {content_type}")
                return '',f"Unexpected content type: {content_type}"
        else:
            # print(f"Failed to fetch data. Status code: {response.status}")
            return '',f"Failed to fetch data. Status code: {response.status}"
    return soup, 'success'
# support function to format context of web page
def  support_formated(text):
    modified_1 = re.sub(r'Contact Details.*', '', text, flags=re.DOTALL)  # to remove last contact details
    modified_2 = re.sub(r'[0-9\n\xA0]',' ',modified_1)  # to remove newline , non breaking space characters
    modified_3  = re.sub(r'[^a-zA-Z\s:.]','',modified_2)
    modified_4  = re.sub(r'\s\.\s+','',modified_3)
    modified_5  = re.sub(r'e.g.','eg',modified_4)
    final = re.sub(r'[:]',' ',modified_5).strip()
    return final
# function to return title and context from url
def get_title_text(url):
    soup , status = support_webdata(url)
    try:
        if status == 'success':
            body_data = soup.body
            text_data = body_data.find('div', class_="td-post-content tagdiv-type")
            context = support_formated(text_data.get_text())
            title = re.sub(r' - Blackcoffer Insights*','.',soup.title.get_text())
            sentence = title + context
            return sentence,status
        else:
            return '',status
    except Exception as e:
        return '',f"error : {e}"


"""
calculate required values using apply 
"""
def call_calculate(r):
    text,status = get_title_text(r['URL'])
    # tokenize words
    word_tokens = np.array(nltk.word_tokenize(text))
    count_words = word_tokens.size
    r['Count Words'] = count_words
    # tokenize sentence
    sentence_tokens = np.array(nltk.sent_tokenize(text))
    count_sent = sentence_tokens.size
    r['Count Sentence'] = count_sent

    #removing stopwords from words
    custom_stopwords_set = set(custom_stopwords)
    puct_words = {"?", "!", ",", "."}
    master_words = np.array([token for token in word_tokens if token.istitle() and token.lower() not in custom_stopwords_set])
    master_words = np.concatenate((master_words ,np.array([token for token in word_tokens if token not in custom_stopwords_set])))
    master_words = np.array([word.lower() for word in master_words if word not in puct_words])

    # Sentiment analysis
    positive_words = np.isin(master_words, predefined_positive_words)
    negative_words = np.isin(master_words, predefined_negative_words)

    # positive score
    positive_score = positive_words.sum()
    # print(positive_score)
    r['POSITIVE SCORE'] = positive_score
    
    # negative score
    negative_score = negative_words.sum()
    # print(negative_score)
    r['NEGATIVE SCORE'] = negative_score
    
    # complex words and syllables
    dic = pyphen.Pyphen(lang='en')
    def get_syllables(word):
        """returns the number of syllables in a given word"""
        syllables = len(dic.inserted(word).split("-"))
        # Adjust syllable count for special case
        if word.endswith(("es", "ed")) and len(word) > 2 and (word[-3] not in "aeiou"):
            syllables -= 1  # Reduce syllable count 
        return syllables 
    
    # Get syllable counts
    syllable_counts = np.array([get_syllables(word) for word in word_tokens])
    
    syllable_count = syllable_counts.sum()
    r['Syllable Count'] = syllable_count
    # complex word count
    complex_word_count = (syllable_counts >= 3).sum()
    r['COMPLEX WORD COUNT'] = complex_word_count
    # word count for master words
    word_count = master_words.size
    r['WORD COUNT'] = word_count
    # personal pronouns
    tags = np.array(nltk.pos_tag(word_tokens))
    personal_pronouns = {word.lower() for word,tag in tags if tag[:3] == 'PRP'}
    r['PERSONAL PRONOUNS'] = personal_pronouns
    # average word length
    avg_word_length = np.mean([len(word) for word in word_tokens])
    r['AVG WORD LENGTH'] = avg_word_length

    return r

# tqdm for pandas
tqdm.pandas()

# apply the function with progress bar
intermediate_df = url_df.progress_apply(call_calculate, axis=1)

output_df = intermediate_df.loc[:,['URL_ID','URL','POSITIVE SCORE','NEGATIVE SCORE']]
positive_scores = intermediate_df['POSITIVE SCORE'].values  
negative_scores = intermediate_df['NEGATIVE SCORE'].values 
epsilon = 1e-6  # To avoid division by zero
polarity_scores = (positive_scores - negative_scores) / (positive_scores + negative_scores + epsilon)
# Assign the polarity scores back to the DataFrame
output_df['POLARITY SCORE'] = polarity_scores
master_word_count = intermediate_df['WORD COUNT'].values
subjective_score = (positive_scores + negative_scores) / (master_word_count + epsilon)
output_df['SUBJECTIVITY SCORE'] = subjective_score
count_words = intermediate_df['Count Words'].values
count_sentences = intermediate_df['Count Sentence'].values
avg_sentence_length = count_words / count_sentences
output_df['AVG SENTENCE LENGTH'] = avg_sentence_length
count_complex_words = intermediate_df['COMPLEX WORD COUNT'].values
percent_complex_words = count_complex_words / count_words
output_df['PERCENTAGE OF COMPLEX WORDS'] = percent_complex_words
fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
output_df['FOG INDEX'] = fog_index
avg_words_per_sentence = count_words / count_sentences
output_df['AVG NUMBER OF WORDS PER SENTENCE'] = avg_words_per_sentence
output_df['COMPLEX WORD COUNT'] = count_complex_words
output_df['WORD COUNT'] = master_word_count
syllable_count = intermediate_df['Syllable Count'].values
syllable_count_per_word = syllable_count / count_words
output_df['SYLLABLE PER WORD'] = syllable_count_per_word
output_df['PERSONAL PRONOUNS'] = intermediate_df['PERSONAL PRONOUNS']
output_df['AVG WORD LENGTH'] = intermediate_df['AVG WORD LENGTH']
output_df.to_excel('output.xlsx', index=False)