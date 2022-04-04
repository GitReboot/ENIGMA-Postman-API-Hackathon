from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import tweepy
import re
import string
import nltk
from unidecode import unidecode
import csv
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')

def replace_sep(text):
  text = text.replace("|||",' ')
  return text

def remove_url(text):
  text = re.sub(r'https?:\/\/.*?[\s+]', '', text)
  return text

def remove_punct(text):
  text=re.sub(r'[^\w\s]', '', text)
  return text

def remove_numbers(text):
  text = re.sub(r'[0-9]', '', text)
  return text

def convert_lower(text):
   text = text.lower()
   return text

def extra(text):
  text=text.replace("  ", " ")
  text=re.sub(r'[^a-zA-Z\s]','',text)
  text=text.strip()
  return text

Stopwords = set(stopwords.words("english"))
def stop_words(text):
  tweet_tokens = word_tokenize(text)
  filtered_words = [w for w in tweet_tokens if not w in Stopwords]
  return " ".join(filtered_words)

def lemmantization(text):
  tokenized_text = word_tokenize(text)
  lemmatizer = WordNetLemmatizer()
  text = ' '.join([lemmatizer.lemmatize(a) for a in tokenized_text])
  return text

def pre_process(text):
    text = replace_sep(text)
    text = remove_url(text)
    text = remove_punct(text)
    text = remove_numbers(text)
    text = convert_lower(text)
    text = extra(text)
    text = stop_words(text)
    text = lemmantization(text)
    return text

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def preproc(s):
    # s=emoji_pattern.sub(r'', s) # no emoji
    s = unidecode(s)
    POSTagger = preprocess(s)
    # print(POSTagger)

    tweet = ' '.join(POSTagger)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in POSTagger:
        if w not in stop_words:
            filtered_sentence.append(w)
    # print(word_tokens)
    # print(filtered_sentence)
    stemmed_sentence = []
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    for w in filtered_sentence:
        stemmed_sentence.append(stemmer2.stem(w))
    # print(stemmed_sentence)

    temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation)
    preProcessed = temp.split(" ")
    final = []
    for i in preProcessed:
        if i not in final:
            if i.isdigit():
                pass
            else:
                if 'http' not in i:
                    final.append(i)
    temp1 = ' '.join(c for c in final)
    # print(preProcessed)
    return temp1


def getTweets(user):
    csvFile = open('user.csv', 'w', newline='')
    csvWriter = csv.writer(csvFile)
    try:
        for i in range(0, 4):
            tweets = api.user_timeline(screen_name=user, count=1000, include_rts=True, page=i)
            for status in tweets:
                tw = preproc(status.text)
                if tw.find(" ") == -1:
                    tw = "blank"
                csvWriter.writerow([tw])
    except tweepy.TweepError:
        print("Failed to run the command on that user, Skipping...")
    csvFile.close()


import tweepy as tw

consumer_key = 'DadKR3DKcG1PWvyh8igvAIaYN'
consumer_secret = 'KjWoOAwm7uwwT0vTGWcuomuPq9Wglo5pA29kPxhOPvddMmO2Eg'
access_token = '1266720191502680066-YGjG1jvAjIOOsG6NibYCDH7trAznfk'
access_token_secret = '1E8fi5w2hi8eRyvnVvHcJnOU9p66oiJaXe5l1PAIQNYqA'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


def join(text):
    return "||| ".join(text)


def twits(handle):
    res = api.user_timeline(screen_name=handle, count=1000, include_rts=True)
    tweets = [tweet.text for tweet in res]
    return tweets


def twit(handle):
    getTweets(handle)
    with open('user.csv', 'rt') as f:
        csvReader = csv.reader(f)
        tweetList = [rows[0] for rows in csvReader]
    with open('newfrequency300.csv', 'rt') as f:
        csvReader = csv.reader(f)
        mydict = {rows[1]: int(rows[0]) for rows in csvReader}

    vectorizer = TfidfVectorizer(vocabulary=mydict, min_df=1)
    x = vectorizer.fit_transform(tweetList).toarray()
    df = pd.DataFrame(x)

    model_IE = pickle.load(open("BNIEFinal.sav", 'rb'))
    model_SN = pickle.load(open("BNSNFinal.sav", 'rb'))
    model_TF = pickle.load(open('BNTFFinal.sav', 'rb'))
    model_PJ = pickle.load(open('BNPJFinal.sav', 'rb'))

    answer = []
    IE = model_IE.predict(df)
    SN = model_SN.predict(df)
    TF = model_TF.predict(df)
    PJ = model_PJ.predict(df)

    b = Counter(IE)
    value = b.most_common(1)

    if value[0][0] == 1.0:
        answer.append("I")
    else:
        answer.append("E")

    b = Counter(SN)
    value = b.most_common(1)

    if value[0][0] == 1.0:
        answer.append("S")
    else:
        answer.append("N")

    b = Counter(TF)
    value = b.most_common(1)

    if value[0][0] == 1:
        answer.append("T")
    else:
        answer.append("F")

    b = Counter(PJ)
    value = b.most_common(1)

    if value[0][0] == 1:
        answer.append("P")
    else:
        answer.append("J")
    mbti = "".join(answer)
    return mbti


def split(text):
    return [char for char in text]


List_jobs_I = ['Accounting manager',
               'landscape designer',
               'Behavioral therapist',
               'Graphic designer',
               'IT manager']

List_jobs_E = ['Flight attendant',
               'Event planner',
               'Teacher',
               'criminal investigator',
               'General manager']

List_jobs_S = ['Home health aide',
               'Detective',
               'Actor',
               'Nurse']

List_jobs_N = ['social worker',
               'HR manager',
               'counselor',
               'Therapist']

List_jobs_F = ['Entertainer',
               'Mentor',
               'Advocate',
               'Artist',
               'Defender',
               'Dreamer']

List_jobs_T = ['Video game designer',
               'Graphic designer',
               'Social media manager',
               'Copywriter',
               'Public relations manager',
               'Digital marketers',
               'Lawyer',
               'Research scientist',
               'User experience designer',
               'Software architect']

List_jobs_J = ['Showroom designer',
               'IT administrator',
               'Marketing director',
               'Judge',
               'Coach']

List_jobs_P = ['Museum curator',
               'Copywriter',
               'Public relations specialist',
               'Social worker',
               'Medical researcher',
               'Office Manager']

List_ch_I = ['Reflective',
             'Self-aware',
             'Take time making decisions',
             'Feel comfortable being alone',
             'Dont like group works']

List_ch_E = ['Enjoy social settings',
             'Do not like or need a lot of alone time',
             'Thrive around people',
             'Outgoing and optimistic',
             'Prefer to talk out problem or questions']

List_ch_N = ['Listen to and obey their inner voice',
             'Pay attention to their inner dreams',
             'Typically optimistic souls',
             'Strong sense of purpose',
             'Closely observe their surroundings']

List_ch_S = ['Remember events as snapshots of what actually happened',
             'Solve problems by working through facts',
             'Programmatic',
             'Start with facts and then form a big picture',
             'Trust experience first and trust words and symbols less',
             'Sometimes pay so much attention to facts, either present or past, that miss new possibilities']

List_ch_F = ['Decides with heart',
             'Dislikes conflict',
             'Passionate',
             'Driven by emotion',
             'Gentle',
             'Easily hurt',
             'Empathetic',
             'Caring of others']

List_ch_T = ['Logical',
             'Objective',
             'Decides with head',
             'Wants truth',
             'Rational',
             'Impersonal',
             'Critical',
             'Firm with people']

List_ch_J = ['Self-disciplined',
             'Decisive',
             'Structured',
             'Organized',
             'Responsive',
             'Fastidious',
             'Create short and long-term plans',
             'Make a list of things to do',
             'Schedule things in advance',
             'Form and express judgments',
             'Bring closure to an issue so that we can move on']

List_ch_P = ['Relaxed',
             'Adaptable',
             'Nonjudgemental',
             'Carefree',
             'Creative',
             'Curious',
             'Postpone decisions to see what other options are available',
             'Act spontaneously',
             'Decide what to do as we do it, rather than forming a plan ahead of time',
             'Do things at the last minute']


def charcter(text):
    o = split(text)
    characteristics = []
    for i in range(0, 4):
        if o[i] == 'I':
            characteristics.append('\n'.join(List_ch_I))
        if o[i] == 'E':
            characteristics.append('\n'.join(List_ch_E))
        if o[i] == 'N':
            characteristics.append('\n'.join(List_ch_N))
        if o[i] == 'S':
            characteristics.append('\n'.join(List_ch_S))
        if o[i] == 'F':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'T':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'J':
            characteristics.append('\n'.join(List_ch_J))
        if o[i] == 'P':
            characteristics.append('\n'.join(List_ch_P))
    crct = '\n'.join(characteristics)
    data = crct.split("\n")
    return data


def recomend(text):
    b = split(text)
    jobs = []
    for i in range(0, 4):
        if b[i] == 'I':
            jobs.append('\n'.join(List_jobs_I))
        if b[i] == 'E':
            jobs.append('\n'.join(List_jobs_E))
        if b[i] == 'N':
            jobs.append('\n'.join(List_jobs_N))
        if b[i] == 'S':
            jobs.append('\n'.join(List_jobs_S))
        if b[i] == 'F':
            jobs.append('\n'.join(List_jobs_F))
        if b[i] == 'T':
            jobs.append('\n'.join(List_jobs_T))
        if b[i] == 'J':
            jobs.append('\n'.join(List_jobs_J))
        if b[i] == 'P':
            jobs.append('\n'.join(List_jobs_P))
    crct1 = '\n'.join(jobs)
    data1 = crct1.split("\n")
    return (split(data1))


def pp(handle):
    personality = twit(handle)
    return personality, recomend(personality), charcter(personality)


from tkinter import *
from PIL import ImageTk
import tkinter as tk


class MyWindow:
    def __init__(self, win):
        self.D_lbl0 = Label(win, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(win, text='Start Application', bg='navy', fg='white', font=("Helvetica", 30),
                           command=self.home1)
        self.btn1.place(x=500, y=300)
        self.btn1 = Button(win, text='Quit', bg='navy', fg='white', font=("Helvetica", 30), command=win.destroy)
        self.btn1.place(x=1000, y=300)

    def mbti(self):
        newwin = Toplevel(window)
        newwin.geometry("1920x1080")
        self.D_lbl0 = Label(newwin, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin, text='MBTI DATA', bg='green', fg='white', font=30,
                           command=lambda: [newwin.destroy(), self.mbti()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin, text='EXPLORATORY DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore(), newwin.destroy()])
        self.btn1.place(x=850, y=120)
        self.D_lbl0 = Label(newwin,
                            text='The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides\n everyone into 16 distinct personality types across 4 axis:\n Introversion (I) — Extroversion (E)\nIntuition (N) — Sensing (S)\nThinking (T) — Feeling (F)\nJudging (J) — Perceiving (P)\nThe dataset contains 8675 observations (people), where each observation gives a person’s:\nMyers-Briggs personality type (as a 4-letter code)\nAn excerpt containing the last 50 posts on their PersonalityCafe forum (each entry separated by “|||”)\nFor example, someone who prefers introversion, intuition, thinking and perceiving would be\n labelled an INTP in the MBTI system, and there are lots of personality based components\n that would model or describe this person’s preferences or behaviour based on the label.\n',
                            fg='black', font=("Helvetica", 25))
        self.D_lbl0.place(x=50, y=230)

    def mbt(self):
        newwin1 = Toplevel(window)
        newwin1.geometry("1920x1080")
        self.D_lbl0 = Label(newwin1, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin1, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin1.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin1, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin1.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin1, text='MBTI TEST', bg='green', fg='white', font=30,
                           command=lambda: [newwin1.destroy(), self.mbt()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin1, text='EXPLORATORY DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore(), newwin1.destroy()])
        self.btn1.place(x=850, y=120)
        canvas = Canvas(newwin1, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="TestResults.png")
        canvas.create_image(1150, 70, image=self.bg1, anchor="ne")

    def explore(self):
        newwin2 = Toplevel(window)
        newwin2.geometry("1920x1080")
        canvas = Canvas(newwin2, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="CountPlot.png")
        canvas.create_image(300, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = Label(newwin2, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin2, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin2.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin2, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin2.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin2, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin2.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin2, text='EXPLORATORY DATA', bg='green', fg='white', font=30,
                           command=lambda: [newwin2.destroy(), self.explore()])
        self.btn1.place(x=850, y=120)
        self.btn1 = Button(newwin2, text='PIE PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore1(), newwin2.destroy()])
        self.btn1.place(x=150, y=250)
        self.btn1 = Button(newwin2, text='DIS PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore2(), newwin2.destroy()])
        self.btn1.place(x=150, y=320)
        self.btn1 = Button(newwin2, text='I-E PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore3(), newwin2.destroy()])
        self.btn1.place(x=150, y=390)
        self.btn1 = Button(newwin2, text='N-S PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore4(), newwin2.destroy()])
        self.btn1.place(x=150, y=460)
        self.btn1 = Button(newwin2, text='T-F PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore5(), newwin2.destroy()])
        self.btn1.place(x=150, y=530)
        self.btn1 = Button(newwin2, text='P-J PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore6(), newwin2.destroy()])
        self.btn1.place(x=150, y=600)

    def explore1(self):
        newwin3 = Toplevel(window)
        newwin3.geometry("1920x1080")
        canvas = Canvas(newwin3, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="PiePlot.png")
        canvas.create_image(300, 20, image=self.bg1, anchor="nw")

        self.D_lbl0 = Label(newwin3, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin3, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin3.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin3, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin3.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin3, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin3.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin3, text='EXPLORATORY DATA', bg='green', fg='white', font=30,
                           command=lambda: [self.explore(), newwin3.destroy()])
        self.btn1.place(x=850, y=120)
        self.btn1 = Button(newwin3, text='PIE PLOT', bg='green', fg='white', font=30,
                           command=lambda: [newwin3.destroy(), self.explore1()])
        self.btn1.place(x=150, y=250)
        self.btn1 = Button(newwin3, text='DIS PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore2(), newwin3.destroy()])
        self.btn1.place(x=150, y=320)
        self.btn1 = Button(newwin3, text='I-E PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore3(), newwin3.destroy()])
        self.btn1.place(x=150, y=390)
        self.btn1 = Button(newwin3, text='N-S PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore4(), newwin3.destroy()])
        self.btn1.place(x=150, y=460)
        self.btn1 = Button(newwin3, text='T-F PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore5(), newwin3.destroy()])
        self.btn1.place(x=150, y=530)
        self.btn1 = Button(newwin3, text='P-J PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore6(), newwin3.destroy()])
        self.btn1.place(x=150, y=600)

    def explore2(self):
        newwin4 = Toplevel(window)
        newwin4.geometry("1920x1080")
        canvas = Canvas(newwin4, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="Displot.png")
        canvas.create_image(300, 20, image=self.bg1, anchor="nw")

        self.D_lbl0 = Label(newwin4, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin4, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin4.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin4, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin4.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin4, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin4.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin4, text='EXPLORATORY DATA', bg='green', fg='white', font=30,
                           command=lambda: [self.explore(), newwin4.destroy()])
        self.btn1.place(x=850, y=120)
        self.btn1 = Button(newwin4, text='PIE PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore1(), newwin4.destroy()])
        self.btn1.place(x=150, y=250)
        self.btn1 = Button(newwin4, text='DIS PLOT', bg='green', fg='white', font=30,
                           command=lambda: [newwin4.destroy(), self.explore2()])
        self.btn1.place(x=150, y=320)
        self.btn1 = Button(newwin4, text='I-E PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore3(), newwin4.destroy()])
        self.btn1.place(x=150, y=390)
        self.btn1 = Button(newwin4, text='N-S PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore4(), newwin4.destroy()])
        self.btn1.place(x=150, y=460)
        self.btn1 = Button(newwin4, text='T-F PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore5(), newwin4.destroy()])
        self.btn1.place(x=150, y=530)
        self.btn1 = Button(newwin4, text='P-J PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore6(), newwin4.destroy()])
        self.btn1.place(x=150, y=600)

    def explore3(self):
        newwin5 = Toplevel(window)
        newwin5.geometry("1920x1080")
        canvas = Canvas(newwin5, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="I_E.png")
        canvas.create_image(500, 100, image=self.bg1, anchor="nw")

        self.D_lbl0 = Label(newwin5, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin5, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin5.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin5, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin5.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin5, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin5.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin5, text='EXPLORATORY DATA', bg='green', fg='white', font=30,
                           command=lambda: [self.explore(), newwin5.destroy()])
        self.btn1.place(x=850, y=120)
        self.btn1 = Button(newwin5, text='PIE PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore1(), newwin5.destroy()])
        self.btn1.place(x=150, y=250)
        self.btn1 = Button(newwin5, text='DIS PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore2(), newwin5.destroy()])
        self.btn1.place(x=150, y=320)
        self.btn1 = Button(newwin5, text='I-E PLOT', bg='green', fg='white', font=30,
                           command=lambda: [newwin5.destroy(), self.explore3()])
        self.btn1.place(x=150, y=390)
        self.btn1 = Button(newwin5, text='N-S PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore4(), newwin5.destroy()])
        self.btn1.place(x=150, y=460)
        self.btn1 = Button(newwin5, text='T-F PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore5(), newwin5.destroy()])
        self.btn1.place(x=150, y=530)
        self.btn1 = Button(newwin5, text='P-J PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore6(), newwin5.destroy()])
        self.btn1.place(x=150, y=600)

    def explore4(self):
        newwin6 = Toplevel(window)
        newwin6.geometry("1920x1080")
        canvas = Canvas(newwin6, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="N_S.png")
        canvas.create_image(500, 100, image=self.bg1, anchor="nw")

        self.D_lbl0 = Label(newwin6, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin6, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin6.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin6, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin6.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin6, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin6.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin6, text='EXPLORATORY DATA', bg='green', fg='white', font=30,
                           command=lambda: [self.explore(), newwin6.destroy()])
        self.btn1.place(x=850, y=120)
        self.btn1 = Button(newwin6, text='PIE PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore1(), newwin6.destroy()])
        self.btn1.place(x=150, y=250)
        self.btn1 = Button(newwin6, text='DIS PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore2(), newwin6.destroy()])
        self.btn1.place(x=150, y=320)
        self.btn1 = Button(newwin6, text='I-E PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore3(), newwin6.destroy()])
        self.btn1.place(x=150, y=390)
        self.btn1 = Button(newwin6, text='N-S PLOT', bg='green', fg='white', font=30,
                           command=lambda: [newwin6.destroy(), self.explore4()])
        self.btn1.place(x=150, y=460)
        self.btn1 = Button(newwin6, text='T-F PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore5(), newwin6.destroy()])
        self.btn1.place(x=150, y=530)
        self.btn1 = Button(newwin6, text='P-J PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore6(), newwin6.destroy()])
        self.btn1.place(x=150, y=600)

    def explore5(self):
        newwin7 = Toplevel(window)
        newwin7.geometry("1920x1080")
        canvas = Canvas(newwin7, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="T_F.png")
        canvas.create_image(500, 100, image=self.bg1, anchor="nw")

        self.D_lbl0 = Label(newwin7, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin7, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin7.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin7, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin7.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin7, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin7.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin7, text='EXPLORATORY DATA', bg='green', fg='white', font=30,
                           command=lambda: [self.explore(), newwin7.destroy()])
        self.btn1.place(x=850, y=120)
        self.btn1 = Button(newwin7, text='PIE PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore1(), newwin7.destroy()])
        self.btn1.place(x=150, y=250)
        self.btn1 = Button(newwin7, text='DIS PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore2(), newwin7.destroy()])
        self.btn1.place(x=150, y=320)
        self.btn1 = Button(newwin7, text='I-E PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore3(), newwin7.destroy()])
        self.btn1.place(x=150, y=390)
        self.btn1 = Button(newwin7, text='N-S PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore4(), newwin7.destroy()])
        self.btn1.place(x=150, y=460)
        self.btn1 = Button(newwin7, text='T-F PLOT', bg='green', fg='white', font=30,
                           command=lambda: [newwin7.destroy(), self.explore5()])
        self.btn1.place(x=150, y=530)
        self.btn1 = Button(newwin7, text='P-J PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore6(), newwin7.destroy()])
        self.btn1.place(x=150, y=600)

    def explore6(self):
        newwin8 = Toplevel(window)
        newwin8.geometry("1920x1080")
        canvas = Canvas(newwin8, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file="J_P.png")
        canvas.create_image(500, 100, image=self.bg1, anchor="nw")

        self.D_lbl0 = Label(newwin8, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin8, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin8.destroy()])
        self.btn1.place(x=590, y=120)
        self.btn1 = Button(newwin8, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin8.destroy()])
        self.btn1.place(x=490, y=120)
        self.btn1 = Button(newwin8, text='MBTI TEST', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbt(), newwin8.destroy()])
        self.btn1.place(x=720, y=120)
        self.btn1 = Button(newwin8, text='EXPLORATORY DATA', bg='green', fg='white', font=30,
                           command=lambda: [self.explore(), newwin8.destroy()])
        self.btn1.place(x=850, y=120)
        self.btn1 = Button(newwin8, text='PIE PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore1(), newwin8.destroy()])
        self.btn1.place(x=150, y=250)
        self.btn1 = Button(newwin8, text='DIS PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore2(), newwin8.destroy()])
        self.btn1.place(x=150, y=320)
        self.btn1 = Button(newwin8, text='I-E PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore3(), newwin8.destroy()])
        self.btn1.place(x=150, y=390)
        self.btn1 = Button(newwin8, text='N-S PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore4(), newwin8.destroy()])
        self.btn1.place(x=150, y=460)
        self.btn1 = Button(newwin8, text='T-F PLOT', bg='navy', fg='white', font=30,
                           command=lambda: [self.explore5(), newwin8.destroy()])
        self.btn1.place(x=150, y=530)
        self.btn1 = Button(newwin8, text='P-J PLOT', bg='green', fg='white', font=30,
                           command=lambda: [newwin8.destroy(), self.explore6()])
        self.btn1.place(x=150, y=600)

    def twitter(self):
        newwin9 = Toplevel(window)
        newwin9.geometry("1920x1080")
        self.D_lbl0 = Label(newwin9, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin9, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin9.destroy()])
        self.btn1.place(x=350, y=120)
        # self.btn1 = Button(newwin9, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        # self.btn1.place(x=750,y=120)
        self.btn1 = Button(newwin9, text='TWITTER POSTS', bg='navy', fg='white', font=30,
                           command=lambda: [self.posts(), newwin9.destroy()])
        self.btn1.place(x=480, y=120)
        self.btn1 = Button(newwin9, text='PREDICT PERSONALITY', bg='navy', fg='white', font=30,
                           command=lambda: [self.home(), newwin9.destroy()])
        self.btn1.place(x=640, y=120)
        self.btn1 = Button(newwin9, text='RECOMMENDATIONS', bg='navy', fg='white', font=30,
                           command=lambda: [self.recomends(), newwin9.destroy()])
        self.btn1.place(x=870, y=120)

    def posts(self):
        newwin10 = Toplevel(window)
        newwin10.geometry("1920x1080")
        self.D_btn1 = Button(newwin10, text='TWITTER POSTS', bg='green', fg='white', font=30,
                             command=lambda: [newwin10.destroy(), self.posts()])
        self.D_btn1.place(x=480, y=120)
        self.D_b1 = Button(newwin10, text='PREDICT PERSONALITY', bg='navy', fg='white', font=30,
                           command=lambda: [self.home(), newwin10.destroy()])
        self.D_b1.place(x=670, y=120)
        self.D_btn1 = Button(newwin10, text='RECOMMENDATIONS', bg='navy', fg='white', font=30,
                             command=lambda: [self.recomends(), newwin10.destroy()])
        self.D_btn1.place(x=930, y=120)
        self.btn1 = Button(newwin10, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin10.destroy()])
        self.btn1.place(x=350, y=120)
        # self.btn1 = Button(newwin10, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        # self.btn1.place(x=750,y=120)
        self.D_lbl0 = Label(newwin10, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.t1 = Text(newwin10)
        self.t3 = Text(newwin10)
        self.t2 = Entry(newwin10, font=150, width=30)
        self.lbl1 = Label(newwin10, text='Enter Twitter ID: ', bg='navy', fg='white', font=("Helvetica", 30))
        self.lbl1.place(x=60, y=230)
        self.lbl4 = Label(newwin10, text='Tweets data of user:', bg='navy', fg='white', font=("Helvetica", 15))
        self.lbl4.place(x=150, y=350)
        self.lbl4 = Label(newwin10, text='Cleaned data:', bg='navy', fg='white', font=("Helvetica", 15))
        self.lbl4.place(x=850, y=350)
        self.t1.place(x=150, y=380)
        self.t3.place(x=850, y=380)
        self.t2.place(x=600, y=230, height=45)
        self.b1 = Button(newwin10, text='Get_Tweets', bg='green', fg='white', font=70, command=self.twt)
        self.b1.place(x=400, y=290, width=130, height=50)
        self.b1 = Button(newwin10, text='PreProcess Tweets', bg='green', fg='white', font=70, command=self.twt1)
        self.b1.place(x=800, y=290, width=170, height=50)

    def twt(self):
        handle = self.t2.get()
        res = twits(handle)
        self.t1.insert(END, str(res))

    def twt1(self):
        handle = self.t2.get()
        res1 = twits(handle)
        tx1 = join(res1)
        tx2 = pre_process(tx1)
        self.t3.insert(END, str(tx2))

    def recomends(self):
        newwin11 = Toplevel(window)
        newwin11.geometry("1920x1080")
        self.D_btn1 = Button(newwin11, text='TWITTER POSTS', bg='navy', fg='white', font=30,
                             command=lambda: [self.posts(), newwin11.destroy()])
        self.D_btn1.place(x=480, y=120)
        self.D_b1 = Button(newwin11, text='PREDICT PERSONALITY', bg='navy', fg='white', font=30,
                           command=lambda: [self.home(), newwin11.destroy()])
        self.D_b1.place(x=670, y=120)
        self.D_btn1 = Button(newwin11, text='RECOMMENDATIONS', bg='green', fg='white', font=30,
                             command=lambda: [newwin11.destroy(), self.recomends()])
        self.D_btn1.place(x=930, y=120)
        self.btn1 = Button(newwin11, text='HOME', fg='white', bg='navy', font=30,
                           command=lambda: [self.home1(), newwin11.destroy()])
        self.btn1.place(x=350, y=120)
        # self.btn1 = Button(newwin11, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        # self.btn1.place(x=750,y=120)
        self.D_lbl0 = Label(newwin11, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.lbl1 = Label(newwin11, text='Enter handle name: ', bg='navy', fg='white', font=("Helvetica", 25))
        self.lbl2 = Label(newwin11, text='Job Recommendations:', bg='navy', fg='white', font=("Helvetica", 15))
        # self.lbl4=Label(text='Characteristics of a person:',bg='navy',fg='white',font=("Helvetica",15))
        self.lbl5 = Label(newwin11, text='Personality Type:', bg='navy', fg='white', font=("Helvetica", 15))
        self.b1 = Button(newwin11, text='Recommendations', bg='green', fg='white', font=40, command=self.recmd)
        self.b1.place(x=700, y=290)
        self.t0 = Entry(newwin11, font=100)
        self.t2 = Text(newwin11, height=15, width=85)
        self.t1 = Entry(newwin11, font=100)
        self.t0.place(x=700, y=220, height=40)
        self.lbl2.place(x=400, y=410)
        self.lbl1.place(x=400, y=220)
        self.lbl5.place(x=500, y=330)
        self.t1.place(x=680, y=330)
        self.t2.place(x=400, y=460)

    def recmd(self):
        handle = self.t0.get()
        res = twit(handle)
        self.t1.insert(END, str(res))
        r = self.t1.get()
        result = recomend(res)
        for i in range(len(result)):
            self.t2.insert(END, str(result[i]))
            self.t2.insert(END, str('\n'))

    def home(self):
        newwin12 = Toplevel(window)
        newwin12.geometry("1920x1080")
        self.D_btn1 = Button(newwin12, text='TWITTER POSTS', bg='navy', fg='white', font=30,
                             command=lambda: [self.posts(), newwin12.destroy()])
        self.D_btn1.place(x=480, y=120)
        self.D_b1 = Button(newwin12, text='PREDICT PERSONALITY', bg='green', fg='white', font=30,
                           command=lambda: [newwin12.destroy(), self.home()])
        self.D_b1.place(x=670, y=120)
        self.D_btn1 = Button(newwin12, text='RECOMMENDATIONS', bg='navy', fg='white', font=30,
                             command=lambda: [self.recomends(), newwin12.destroy()])
        self.D_btn1.place(x=930, y=120)
        self.btn1 = Button(newwin12, text='HOME', bg='navy', fg='white', font=30,
                           command=lambda: [self.home1(), newwin12.destroy()])
        self.btn1.place(x=350, y=120)
        # self.btn1 = Button(newwin12, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        # self.btn1.place(x=750,y=120)
        self.D_lbl0 = Label(newwin12, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.lbl2 = Label(newwin12, text='Characteristics of Personalities:', bg='navy', fg='white',
                          font=("Helvetica", 25))
        self.lbl2.place(x=830, y=250)
        self.t = Text(newwin12, height=15, width=85)
        self.t.place(x=830, y=310)
        self.lbl2 = Label(newwin12, text='Predicted Personality Type ', bg='navy', fg='white', font=("Helvetica", 25))
        self.lbl2.place(x=30, y=430)
        self.lbl1 = Label(newwin12, text='Enter handle name of twitter: ', bg='navy', fg='white',
                          font=("Helvetica", 25))
        self.lbl1.place(x=30, y=300)
        self.t1 = Entry(newwin12, bd=3, font=100)
        self.t1.place(x=480, y=300, height=40)
        self.t2 = Entry(newwin12, bd=3, font=100)
        self.t2.place(x=480, y=430, height=40)
        self.b1 = Button(newwin12, text='Predict_personality', bg='green', fg='white', font=70, command=self.predict)
        self.b1.place(x=400, y=380)

    def predict(self):
        handle = self.t1.get()
        res = twit(handle)
        self.t2.insert(END, str(res))
        r = self.t2.get()
        result = charcter(res)
        for i in range(len(result)):
            self.t.insert(END, str(result[i]))
            self.t.insert(END, str('\n'))

    def home1(self):
        newwin13 = Toplevel(window)
        newwin13.geometry("1920x1080")
        self.bg1 = ImageTk.PhotoImage(file="Home.png")
        canvas = Canvas(newwin13, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        canvas.create_image(1250, 70, image=self.bg1, anchor="ne")
        self.D_lbl0 = Label(newwin13, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(newwin13, text='MBTI DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.mbti(), newwin13.destroy()])
        self.btn1.place(x=550, y=120)
        self.btn1 = Button(newwin13, text='TWITTER DATA', bg='navy', fg='white', font=30,
                           command=lambda: [self.posts(), newwin13.destroy()])
        self.btn1.place(x=750, y=120)


window = tk.Tk()
window.title("Main Screen")
mywin = MyWindow(window)
window.geometry("1920x1080")
window.mainloop()
