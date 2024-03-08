import nltk
from nltk.tokenize import word_tokenize
from googletrans import Translator
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import emoji
import cleantext
def cleaned_text(text):
    text= emoji.replace_emoji(text, replace='')
    return cleantext.clean_words(text,stemming=False,)

nltk.download(['punkt','stopwords','twitter_samples'])
removeables = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3', ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';(', '(', ')','*','=','!',"'",'&amp;',',',':','.','-','_','0','1','2','3','4','5','6','7','8','9',
    'via','RT','\n','#','@','http'}

tweets = [[1,t] for t in nltk.corpus.twitter_samples.strings('positive_tweets.json')]
tweets1 = [[-1,t] for t in nltk.corpus.twitter_samples.strings('negative_tweets.json')]
tweets=tweets+tweets1

def give_emoji_free_text(text):
    return emoji.replace_emoji(text, replace='')

for i in range(0,len(tweets)):
    tweets[i][1]=give_emoji_free_text(tweets[i][1])


for i in range(0,len(tweets)):
    for t in removeables:
        if t=='#':
             pattern = r'\#\w+'
             tweets[i][1] = re.sub(pattern, '', tweets[i][1]).strip().lower()
        elif t=='@':
            pattern = r'\@\w+'
            tweets[i][1] = re.sub(pattern, '', tweets[i][1]).strip().lower()
        elif t=='http':
            pattern = r'http\S+'
            tweets[i][1] = re.sub(pattern, '', tweets[i][1]).strip().lower()
        elif t in tweets[i][1]:
            tweets[i][1]=tweets[i][1].replace(t,"").strip().lower()




def tokenize_text(text):
    return word_tokenize(text)

# Stopwords removal
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

labels=[]
# Apply feature extraction functions to each text in the dataset
processed_text_data = []
for text in tweets:
    labels.append(text[0])
    tokens = tokenize_text(text[1])
    tokens = remove_stopwords(tokens)
    processed_text_data.append(tokens)


for i in range(len(processed_text_data)):
    temp=""
    for j in processed_text_data[i]:
        temp=temp+ ' '+j
    tweets[i][1]=temp


X = [" ".join(tokens) for tokens in processed_text_data]  # Convert tokenized text data into strings
y = labels  # Replace labels_list with your actual labels

# Initialize CountVectorizer to convert tokenized text into feature vectors
vectorizer = CountVectorizer()

# Fit and transform the tokenized text into feature vectors
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the sentiment analysis model 
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_data = [
    "I absolutely loved the movie! The acting was fantastic and the storyline was gripping.",
    "The customer service was terrible. I had to wait on hold for over an hour and the representative was rude.",
    "The food at the restaurant was delicious, but the service was slow and the prices were high.",
    "I'm so excited to start my new job next week! I can't wait to meet my coworkers and dive into the work.",
    "I'm feeling really stressed out lately. There's so much going on at work and I'm having trouble keeping up.",
    "The weather is perfect today! It's sunny and warm, with a gentle breeze.",
    "I'm disappointed with the quality of this product. 12It didn't live up to the advertised features.",
    "pa,super machi",
    "itna bura tha zindagi me never again",
    "waste movie bro thiruppi varave maaten"
]


translator=Translator()
translated=[]
for i in new_data:
    trans = translator.translate(i)
    translated.append(cleaned_text(trans.text))
translated=[re.sub(r"['\[\],]", "", str(t)).strip() for t in translated]
print(translated)

new_input_features = vectorizer.transform(translated)
# Predict the sentiment of the new inputs using the trained model
predicted_sentiments = model.predict(new_input_features)
print(predicted_sentiments)