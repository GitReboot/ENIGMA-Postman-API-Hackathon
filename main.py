import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go

import re
from wordcloud import WordCloud
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Data Import
data = pd.read_csv('mbti_1dataset.csv')
data.head(10)
data.tail(10)
data.info()
data.isnull().sum()
data.describe()

# Exploratory Data Analysis
df1 = data.copy()

Personality_types = df1['type'].unique()
print(Personality_types)
count = df1.groupby(['type']).count()


def plot(data):
    plt.figure(figsize=(20, 7))
    plt.xticks(fontsize=24, rotation=0)
    plt.yticks(fontsize=24, rotation=0)
    return sns.countplot(data=df1, x='type')


plot(df1)

count = df1.groupby(['type'])['posts'].count()
pie = go.Pie(labels=count.index, values=count.values)
figure = go.Figure(data=[pie])
py.iplot(figure)

df1["NumPosts"] = df1["posts"].apply(lambda x: len(x.split("|||")))

s1 = df1["NumPosts"].unique()
s1.sort()
print(s1)

count = df1.groupby(['NumPosts']).count()
print(count[30:53])

plt.figure(figsize=(35, 10))
sns.displot(data=df1, x='NumPosts')


def extract_urls(text):
    urls = re.findall(r'(https?:\/\/.*?[\s+])', text)
    return ",".join(urls)


df1['url_links'] = df1['posts'].apply(lambda x: extract_urls(x))


def countyoutubelinks(posts):
    count = 0
    for p in posts:
        if 'youtube' in p:
            count += 1
    return count


df1['youtube_links'] = df1['url_links'].apply(lambda x: x.strip().split(",")).apply(countyoutubelinks)

map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df1['I-E'] = df1['type'].astype(str).str[0]
df1['I-E'] = df1['I-E'].map(map1)
df1['N-S'] = df1['type'].astype(str).str[1]
df1['N-S'] = df1['N-S'].map(map2)
df1['T-F'] = df1['type'].astype(str).str[2]
df1['T-F'] = df1['T-F'].map(map3)
df1['J-P'] = df1['type'].astype(str).str[3]
df1['J-P'] = df1['J-P'].map(map4)

len(df1[df1['I-E'] == 0])
len(df1[df1['I-E'] == 1])
len(df1[df1['N-S'] == 0])
len(df1[df1['N-S'] == 1])
len(df1[df1['T-F'] == 0])
len(df1[df1['T-F'] == 1])
len(df1[df1['J-P'] == 0])
len(df1[df1['J-P'] == 1])

personalities = df1.loc[:, "I-E":"J-P"].columns
for personality in personalities:
    sns.countplot(x=df1[personality], data=df1)
    plt.show()

df_by_personality = df1.groupby("type")['posts'].apply(' '.join).reset_index()


def generate_wordcloud(text, title):
    wordcloud = WordCloud(background_color="white", width=400, height=300).generate(text)
    plt.subplots(1, 1, figsize=[20, 6])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=40)
    plt.show()


for i, t in enumerate(df_by_personality['type']):
    text = df_by_personality.iloc[i, 1]
    generate_wordcloud(text, t)

# Pre Processing

pd.set_option('display.max_colwidth', 1000)


def replace_sep(text):
    text = text.replace("|||", ' ')
    return text


def remove_url(text):
    text = re.sub(r'https?:\/\/.*?[\s+]', '', text)
    return text


def remove_punct(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_numbers(text):
    text = re.sub(r'[0-9]', '', text)
    return text


def convert_lower(text):
    text = text.lower()
    return text


def extra(text):
    text = text.replace("  ", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
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


data['cleaned_posts'] = data['posts'].apply(lambda x: pre_process(x))

data.head(5)

# Training the model

df = data

# Training Data
vector = CountVectorizer(analyzer="word",
                         max_features=1500,
                         tokenizer=None,
                         preprocessor=None,
                         stop_words=None,
                         max_df=0.7,
                         min_df=0.1)
X = vector.fit_transform(df.cleaned_posts)
Y = np.array(df.type)

tfidf_transformer = TfidfTransformer()
X_final = tfidf_transformer.fit_transform(X)

feature_names = list(enumerate(vector.get_feature_names()))

print(Y.shape)
print(X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.3, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_predictiontrain = random_forest.predict(X_train)
Y_predictiontest = random_forest.predict(X_test)

print("Train Accuracy:", np.mean(Y_predictiontrain == Y_train))
print("Test Accuracy:", np.mean(Y_predictiontest == Y_test))

accuracy_randomforest = (random_forest.score(X_train, Y_train) * 100)
print(accuracy_randomforest, "%")

from sklearn.svm import SVC

svm = SVC(kernel='linear')
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.3, random_state=42)
svm.fit(X_train, Y_train)

Y_predictiontrain = svm.predict(X_train)
Y_predictiontest = svm.predict(X_test)

print("Train Accuracy:", np.mean(Y_predictiontrain == Y_train))
print("Test Accuracy:", np.mean(Y_predictiontest == Y_test))

svm.score(X_train, Y_train)
acc_svm = round(svm.score(X_train, Y_train) * 100, 2)
print(round(acc_svm, 2, ), "%")

# # Testing

d3 = data[8635:8675].copy()

d3.shape


def identify_personality(test):
    text = tfidf_transformer.transform(vector.transform([test])).toarray()
    all_words = vector.get_feature_names()
    t1 = pd.DataFrame.from_dict({words: text[:, i] for i, words in enumerate(all_words)})
    x = random_forest.predict(t1)
    return x


d3['predicted_personality'] = d3['cleaned_posts'].apply(lambda x: identify_personality(x))

# # Validation

# sample-1

data[8670:8671]

q1 = "ixfp always think cat fi doms reason especially website become neo nazi perc im nerd ive learning dutch duolingo im much fun duolingo shit oh god love xd right winger lack political consciousness doubt real hope hell theyre nothing like twilight vampire would agree likely related confidence level make sense someone would aggressive overcompensate sometimes perceived shortcoming nazi germany soviet union extremely nationalist cant think nationalist country nowadays dprk separate nationalism youre one plenty infps right wing conservativesalt reich fascist perc time zone patrick walker voice angel eh cloggies pretty cool people experience feel like going gym morning went anyway one best workout anyway dont feel like exercising anyway youll glad actually thought brand cologne first heard name laughing apparently tempbanned lot people forum theyre pretty dead late color ive come realize need spend time working improving skill learning new thing im usually lazy sitting around watching youtube playing video game dont felt like good time get fresh new name bit profile makeover george soros gave lot money kid contribute white genocide muslim arab move turn u caliphate im part honestly idea even since shahada gone see youre trying start personality cult replace right poly many tic blood sucking parasite youre welcome asking real question choice fuck either two men must choose one cant neither look exactly feel free use imagination please change name zeta neprok im tired old name thanks dont dissin bob ross eat pineapple pizza would leave embarrassment younger self thats youre smart fuck shahada always nice admit found sort confusing always odds crucial issue obviously shitpost thread stereotype ridiculous granted probably fit lot art skill horrendous always got ta love people use mbti stroke ego rolleyes im going say stupidity think people stupid anything like know many time nuclear war almost happened computer glitch somehow managed get coffee date weekend probably miracle sort im pretty drunk night hasnt even started yet laughing free market isnt free fact extremely coercive authoritarian reason state state institution influenced public sometimes working class people push reform anything feel comfortable around woman around men perc become neo nazi im pretty stupid whats interesting dont talk much lot people dont hear say stupid shit think im smart sucker laughing although disagree said multiculturalism assuming understood meant everything else said right made lot good point dont think he right say thats end goal multiculturalism would agree goal remove white dominance western society let honest thats course allowed sex white people otherwise youre sjw liberal cuck white child contributing white genocide also moderate even mean change depending context political situation time moderate would seen radical extremist people today would"

q2 = tfidf_transformer.transform(vector.transform([q1])).toarray()

all_words = vector.get_feature_names()
q3 = pd.DataFrame.from_dict({words: q2[:, i] for i, words in enumerate(all_words)})

x = []
x = random_forest.predict(q3)
y = x[0]

print(y)

# sample-2

data[8674:8675]

p1 = "long since personalitycafe although doesnt seem changed one bit must say good back somewhere like usually turn doctor overwhelmed world around one dream chased large shadowy creature someone else felt save else dream ended reached safety happened well avatar doctor clockwork creature always liked monster worker trying job kind st thanks reply appreciate help get nd think everyone right opinion however many people abuse right p yea iron man thing xd thanks advice everyone thanks think needed humour might show maybe know wont anything harsh like throw beat although place go really dont dont really know judge personality type say mum introverted dad extroverted p ok understand want feel liking men thats good problem tell parent want tell really dont know say im contempt always suppose spent afternoon buried book world spending morning animal suppose good mix whats hardest thing im going face life havent got clue current perspective telling parent im gay seems like huge hurdle suspect seem mi torchwood oh thats fictional oh well still got book two day ago read thought brilliant true family love never wanted deeply understand contempt keeping deeper self different people see look understand knowing important much know also happy let anything get lately feel better either guess wasnt watching comedy central incredible song thanks reply relate part least big deal may made reply pointing well something along line feel although ever entered state mind time need emotion sake sometimes think im young time live life find parent never happy effort give anything say used quite lazy lately tried best everything ask without hassle hey spoiler button still preload image doesnt matter im best driver currently hold restricted licence drive safely dont tell parent speed open road one around dont let someone tell belief wrong think right true people may agree make belief wrong mean midnight blue question many thing feel im contrasting may see dark shade blue fact story behind every line magnificent im fine ad make website know sometimes need reccomendation though make open new window clicked targettop yes want people understand people anyway think doesnt matter could understand many people dont care really fully interested know infjs idea letting people life keeping distance personaly people let world let one wouldnt bothered personally actually sometimes actually think dont realize obvious others might feel dont see good someone wont hang around happens chance wont notice accident good disappearing dont want see library walk somewhere remote alone agree although im black white person outside anyway website need colour bright black doesnt help im defiantly night person doubt enjoy late sleeping half day away say lost balance knew exactly feel say im similar place right lost thing really keep going balance world spinning look good think top navigation barnews would look better darker colour white seems bright would tenth way actually capt jack going say doctor thought second thing seemed appropriate apart tardis would get car drive stopping enjoy thanks solitude right like constructive criticism something work improve guess upset little feel people criticize work something whenever someone tell something done good enough make feel like good enough maybe yes owner lonely heart doctor fan say repeated memory wipe silence eventually fry brain eleventh doctor im remember people currently reading white wolf son good read quite expected picked finish soon next want read wolf gift tonight sit outside window blackness night staring wilderness call space peace beauty could stay till morning light thousand sun stare upon going close facebook month back well wanting able message family ausse school friend found connected website second mar collection seems fitting mood right seen agree actually think first time watched movie beginning got power kinda thought andrew would never work right ok watched underworld awakening must say really good film compared film last month anyway dont think good first would never want turn emotion sometimes hide world still need"

p2 = tfidf_transformer.transform(vector.transform([p1])).toarray()

all_words = vector.get_feature_names()
p3 = pd.DataFrame.from_dict({words: p2[:, i] for i, words in enumerate(all_words)})

x = []
x = random_forest.predict(p3)
y = x[0]

print(y)

# sample-3

data[8673:8674]

test = "conflicted right come wanting child honestly maternal instinct whatsoever recently none close friend child guess closest friend isfj esfp istj xnfp esfj dont know correct dont know know type actually xnfp said last paragraph teacher frustrates there trend education combine class contain variety type student wont cant say much trouble community general mean im talking sameage peer never considered popular artsy extracurricular eat meat feel guilty vaguely considered eating meat dont think could actually nutritional requirement need high protein diet well boat except teach science theater know people dont think science infp find interesting im good another career exactly god truly created one man one woman technically created woman man product incest per book stayed list going around facebook choice following special order reader bernhard schlink interpreter malady jhumpa amazing list im partially posting im sure find need idea listen list isnt long others im pretty another point infp type known idealist heard theory procrastination say come person setting high standard mind ideal perhaps well isnt bit impulse control im feeling calm one else going feel calm cute dont think stereotype think eleanor roosevelt may infp google classified enfp read biography actually seems like introvert forced dont know much subtypes wow definitely teacher first time read thought yeah way reading carefully second time almost perfectly list interest hobby generally like read lot ill tell later though sometimes work get busy also enjoy writing come burst go well cant help heisnt gay thing sorry however commiserate part waiting someone text back feeling like power oh wow totally situation right im treated poorly opposite problem cant seem find courage stand ground dont think cry much mom think im pretty sure think im clinically depressed even though im mean im pretty sure im cry bit lol well feel dumb work see thanks alcohol doesnt actually change base personality easy change mbti wouldnt valid alcohol make say thing would absent site couple month since ive back couple week ago whenever try create thread like right reply thread unable create new also sorry huge long single paragraph reason window wont let create new paragraph attempting fix eggsies thanks reply perspective cheating thing interesting never really thought way especially beginning perhaps make good point infp forum conversation like getting gift agreed lot thing however im terrible gift giver anyone different taste im always good math never enjoyed math fair im slightly f opposed function also say math methodical ok mr enfj conundrum dating enfj since january busy schedule able spend little time love bike use one recreation actual transportation might bike weekend evening errand weather nice except right live steep hill id never make havent clinically diagnosed adhd reading extensively talking mom behavior kid found wanted take psychologist dad yoga clothes tight black capri pink purple tank tear movie sometimes dont generally cry hard exception movie version book read like harry potter believe two hour usually unfortunately kind read lot book fiction im working night circus erin morgenstern almost done dovekeepers alice hoffman also reading im little ambivalent book made movie many favorite book literary enough movie version would atrocious couple agree finish dark could read point without replying may physically alone someone hearing havent posted thread fear may belong curiosity someone desperately wanted meet need didnt know would tell person would even know ask well ive never read book seen looselybased movie see youre saying thought im fully convinced case im greatly appreciate input boxerkitty said enfjs need socialize wide set people time unwind make feel somewhat better one thing would love person choose spend life assuming find someone choose spend life similar point made completely agreed madrabbit exception im pursuing answer never find possibly doesnt exist go cycle get frustrated give decide stop yeah hear know really complicated wish seems way mind one dont entirely know bring without using word marriage dont know sure enfj seems far likely thought id post advice enfjs difficulty relationship something bugging week thought would post people seem get think thing way im seeing original post title anything go totally relate almost posted similar thread able trust relationship day ago desire control others kind freak feel like control feeling situation let explain desire tell someone else others mentioned feel really guilty spend lot money spatype thing feel like could putting somewhere useful especially since dont make ton sad angry pathetic embarrassed love oh definitely definitely stem combination number thing occurred life biggest overall factor form attachment incredibly easy seems im really hoping since referred breakup letter longdistance relationship seriously cant even face person tell feel cant honestly see point saying letter id like change cant imagine anyone said voted intp im closer e ok thing sort issue test took quiz got entertainer one dont think really type reason ill describe two yeah definitely get jealous come romantic relationship honest havent many serious romantic relationship part reason though certainly wait leafstone third volume q book originally split japanese another language spent entire week beach book didnt finish finished since"

text = tfidf_transformer.transform(vector.transform([test])).toarray()

all_words = vector.get_feature_names()
t1 = pd.DataFrame.from_dict({words: text[:, i] for i, words in enumerate(all_words)})

x = []
x = random_forest.predict(t1)
y = x[0]

print(y)

# # Testing data using Twitter data

import tweepy as tw

consumer_key = 'DadKR3DKcG1PWvyh8igvAIaYN'
consumer_secret = 'KjWoOAwm7uwwT0vTGWcuomuPq9Wglo5pA29kPxhOPvddMmO2Eg'
access_token = '1266720191502680066-YGjG1jvAjIOOsG6NibYCDH7trAznfk'
access_token_secret = '1E8fi5w2hi8eRyvnVvHcJnOU9p66oiJaXe5l1PAIQNYqA'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

handle = '@sundarpichai'

res = api.user_timeline(screen_name=handle, count=50, include_rts=True)
tweets = [tweet.text for tweet in res]


def join(text):
    return "||| ".join(text)


t1 = join(tweets)
t2 = pre_process(t1)
t3 = tfidf_transformer.transform(vector.transform([t2])).toarray()
all_words = vector.get_feature_names()
t4 = pd.DataFrame.from_dict({words: t3[:, i] for i, words in enumerate(all_words)})

x = random_forest.predict(t4)

print(x)


def twits(handle):
    res = api.user_timeline(screen_name=handle, count=50, include_rts=True)
    tweets = [tweet.text for tweet in res]
    return tweets


def twit(handle):
    res = api.user_timeline(screen_name=handle, count=50, include_rts=True)
    tweets = [tweet.text for tweet in res]
    t1 = join(tweets)
    t2 = pre_process(t1)
    t3 = tfidf_transformer.transform(vector.transform([t2])).toarray()
    all_words = vector.get_feature_names()
    t4 = pd.DataFrame.from_dict({words: t3[:, i] for i, words in enumerate(all_words)})
    x = []
    x = random_forest.predict(t4)

    y = x[0]
    return y


twit('@finkd')

a = twit('@BillGates')

print(a)

b = twit('@sundarpichai')

print(b)


# # Recommendations

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
               'Criminal investigator',
               'General manager']

List_jobs_S = ['Home health aide',
               'Detective',
               'Actor',
               'Nurse']

List_jobs_N = ['Social worker',
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
             'Non-judgemental',
             'Carefree',
             'Creative',
             'Curious',
             'Postpone decisions to see what other options are available',
             'Act spontaneously',
             'Decide what to do as we do it, rather than forming a plan ahead of time',
             'Do things at the last minute']


def character(text):
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
    return split(data1)


def pp(handle):
    personality = twit(handle)
    return personality, recomend(personality), character(personality)


character(twit('@finkd'))

recomend(twit('@finkd'))

# # user_interface

from tkinter import *
from PIL import ImageTk
import tkinter as tk


class MyWindow:
    def __init__(self, win):
        self.D_lbl0 = Label(win, text='Personality Based Job Recommender Using Twitter Data ', fg='navy',
                            font=("Helvetica", 40))
        self.D_lbl0.place(x=110, y=30)
        self.btn1 = Button(win, text='Start Application', bg='navy', fg='white', font=("Helvetica", 30), command=self.home1)
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
        result = character(res)
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
