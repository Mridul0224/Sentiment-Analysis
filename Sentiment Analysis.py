#!/usr/bin/env python
# coding: utf-8

# # Read in Data and NLTK Basics

# In[257]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[258]:


#Read in data
df = pd.read_csv(r"C:\Users\mridu\OneDrive\Desktop\Portfolio Projects\Sentiment Analysis\Reviews.csv")
print(df.shape)
df = df.head(500)
print(df.shape)



# In[259]:


df['Text'].values[0]
#df['review'].iloc[0]


# In[260]:


df.head()


# ## EDA

# In[261]:


ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()


# ## Basic NLTK

# In[262]:


example = df['Text'][50]
print(example)


# In[263]:


#nltk.download('punkt')
tokens = nltk.word_tokenize(example)
tokens[:10]


# In[264]:


#nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[265]:


#nltk.download('words')
#nltk.download('maxent_ne_chunker')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# # VADER Seniment Scoring
# 
# We will use NLTK's `SentimentIntensityAnalyzer` to get the neg/neu/pos scores of the text.
# 
# - This uses a "bag of words" approach:
#     1. Stop words are removed
#     2. each word is scored and combined to a total score.

# In[266]:


nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[267]:


sia.polarity_scores('I am so happy!')


# In[268]:


sia.polarity_scores('This is the worst thing ever.')


# In[269]:


sia.polarity_scores(example)


# In[270]:


# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[271]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[272]:


# Now we have sentiment score and metadata
vaders.head()


# ## Plot VADER results

# In[273]:


ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()


# In[274]:


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# # Roberta Pretrained Model
# 
# - Use a model trained of a large corpus of data.
# - Transformer model accounts for the words but also the context related to other words.

# In[275]:


#!pip install torch
#!pip install transformers
#!pip install scipy
#!pip install spicy


# In[276]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[277]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[278]:


# VADER results on example
print(example)
sia.polarity_scores(example)


# In[279]:


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[280]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[281]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[284]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# ## Compare Scores between models

# In[285]:


results_df.columns


# # Combine and compare

# In[286]:


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


# # Output Examples:
# 
# - Positive 1-Star and Negative 5-Star Reviews
# 
# Lets look at some examples where the model scoring and review score differ the most.

# In[287]:


results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[288]:


results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[289]:


# nevative sentiment 5-Star view


# In[290]:


results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]


# In[291]:


results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]


# # Transformers Pipeline

# In[292]:


from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


# In[314]:


sent_pipeline('how are you today!')


# In[315]:


sent_pipeline('I am crying')


# In[316]:


sent_pipeline('it was kind of okay')


# # BERT Sentiment Analysis

# ## Install and Import Dependencies

# In[43]:


get_ipython().system('pip install transformers requests beautifulsoup4 pandas numpy')


# In[296]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


# ## Instantiate Model

# In[297]:


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# ## Encode and Calculate Sentiment

# In[298]:


tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')
tokens


# In[299]:


result = model(tokens)


# In[300]:


result.logits


# In[ ]:


int(torch.argmax(result.logits))+1


# ## Collect Reviews

# In[302]:


r = requests.get('https://www.yelp.com/biz/gordon-ramsay-hell-s-kitchen-mashantucket-2')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]


# In[303]:


reviews


# ## Load Reviews into DataFrame and Score

# In[304]:


import numpy as np
import pandas as pd


# In[305]:


df = pd.DataFrame(np.array(reviews), columns=['review'])


# In[306]:


df['review'].iloc[0]


# In[307]:


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[308]:


sentiment_score(df['review'].iloc[1])


# In[309]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))


# In[310]:


df


# ## Visualization

# In[311]:


ax = sns.barplot(data=df, x='review', y='sentiment')
ax.set_title('Bert Sentiment Analysis')
plt.figure(figsize=(10, 8))
plt.show()

import matplotlib.pyplot as plt

# Set the size of your plot for better visibility
plt.figure(figsize=(10, 8))

# Create the bar chart
plt.bar(df['review'], df['sentiment'])

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Adding labels and title (optional but recommended for clarity)
plt.xlabel('Review ID')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of Reviews')

# Show the plot
plt.tight_layout()  # This adjusts subplot params to give some padding.
plt.show()


# In[313]:


df['review'].iloc[3]


# # Comparison Between VADER, ROBERTA AND BERT

# In[222]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['review']
    res[text] = sia.polarity_scores(text)


# In[223]:


data = df
df = pd.DataFrame(data)

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    # Accessing the first column of the DataFrame where the review text is located
    text = row.iloc[0]  # Changed from row['review'] to row.iloc[0]
    res[text] = sia.polarity_scores(text)

# If you want to see the result
print(res)


# ## Calculating Vader for the scrapped Data

# In[224]:


df_res = pd.DataFrame.from_dict(res, orient='index').reset_index()

# Rename the columns to reflect their contents
df_res.columns = ['review', 'neg', 'neu', 'pos', 'compound']
df_res.rename(columns={'neg': 'Vaders_neg', 'neu': 'Vaders_neu', 'pos': 'Vaders_pos'}, inplace=True)

# Display the resulting DataFrame
print(df_res)


# In[225]:


# Assuming df contains a column named 'sentiment' and df_res contains the sentiment scores along with 'review'
# Merge df_res with df to add the 'sentiment' column to df_res based on the 'review' column
merged_df = pd.merge(df_res, df[['review', 'sentiment']], on='review', how='left')

# Display the resulting merged DataFrame
print(merged_df)


# ## Plotting new Vader data

# In[251]:


fig, axs = plt.subplots(1, 3, figsize=(10, 8))
sns.barplot(data=merged_df, x='review', y='Vaders_pos', ax=axs[0])
sns.barplot(data=merged_df, x='review', y='Vaders_neu', ax=axs[1])
sns.barplot(data=merged_df, x='review', y='Vaders_neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# ## Calculating Roberta for the scrapped Data

# In[234]:


def polarity_scores_roberta(example):
    try:
        encoded_text = tokenizer(example, return_tensors='pt', max_length=512, truncation=True)
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        # Return a default dictionary in case of error
        return {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}


# In[237]:


print(scores)


# In[244]:


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Your provided polarity_scores_roberta function
def polarity_scores_roberta(merged_df):
    try:
        encoded_text = tokenizer(merged_df, return_tensors='pt', max_length=512, truncation=True)
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        # Return a default dictionary in case of error
        return {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}

    return scores

def sentiment_score(review):
    # Ensure the review is truncated to the model's max input size
    tokens = tokenizer.encode(review, return_tensors='pt', truncation=True, max_length=512)
    result = model(tokens)
    return int(torch.argmax(result.logits))+1



# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Assuming merged_df is your DataFrame and it contains a column named 'review'
# Initialize an empty list to store combined results
combined_results = []

# Iterating through each row in merged_df to calculate sentiment scores from both models
for i, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    text = row['review']  # or row.iloc[0] if the column is unnamed
    vader_result = sia.polarity_scores(text)
    roberta_result = polarity_scores_roberta(text)
    sentiment_score_result = sentiment_score(text)
    combined_result = {'sentiment_score_result': sentiment_score_result, **vader_result, **roberta_result, 'review': text}
    combined_results.append(combined_result)

# Convert the combined results into a DataFrame
results_df = pd.DataFrame(combined_results)

# Display the results DataFrame
print(results_df.head())


# In[245]:


results_df.columns


# ## Comparison Visualization

# In[255]:


# Categorizing sentiment_score_result
results_df['custom_neg'] = results_df['sentiment_score_result'].apply(lambda x: 1 if x <= 2 else 0)
results_df['custom_neu'] = results_df['sentiment_score_result'].apply(lambda x: 1 if x == 3 else 0)
results_df['custom_pos'] = results_df['sentiment_score_result'].apply(lambda x: 1 if x >= 4 else 0)


# In[256]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data selection - let's select the first 5 reviews for simplicity
sample_df = results_df.head(5)

# Indices for plotting
indices = np.arange(len(sample_df))

# Width of the bars
bar_width = 0.15

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

# Plotting for each model
ax.bar(indices, sample_df['neg'], bar_width, label='VADER Negative')
ax.bar(indices + bar_width, sample_df['neu'], bar_width, label='VADER Neutral')
ax.bar(indices + 2*bar_width, sample_df['pos'], bar_width, label='VADER Positive')

ax.bar(indices + 3*bar_width, sample_df['roberta_neg'], bar_width, label='RoBERTa Negative')
ax.bar(indices + 4*bar_width, sample_df['roberta_neu'], bar_width, label='RoBERTa Neutral')
ax.bar(indices + 5*bar_width, sample_df['roberta_pos'], bar_width, label='RoBERTa Positive')

ax.bar(indices + 6*bar_width, sample_df['custom_neg'], bar_width, label='Custom Negative')
ax.bar(indices + 7*bar_width, sample_df['custom_neu'], bar_width, label='Custom Neutral')
ax.bar(indices + 8*bar_width, sample_df['custom_pos'], bar_width, label='Custom Positive')

# Labels, title, and legend
ax.set_xlabel('Reviews')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Sentiment Analysis Methods Across Reviews')
ax.set_xticks(indices + 4*bar_width)
ax.set_xticklabels(sample_df['review'].apply(lambda x: x[:10] + '...'), rotation=45, ha="right")
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()


# In[ ]:




