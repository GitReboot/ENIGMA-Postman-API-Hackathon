#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[43]:


get_ipython().system('pip install tweepy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install plotly')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install nltk')
get_ipython().system('pip install sklearn')


# In[49]:


import os 
import tweepy as tw
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[50]:


nltk.download('punkt')


# In[51]:


nltk.download('wordnet')


# In[52]:


nltk.download('stopwords')


# # Data Import

# In[53]:


data=pd.read_csv('mbti_1dataset.csv')


# In[54]:


data.head(10)


# In[55]:


data.tail(10)


# In[56]:


data.shape


# In[57]:


data.info()


# In[58]:


data.isnull().sum()


# In[59]:


data.describe()


# # Exploratory Data Analysis

# In[60]:


df1=data.copy()


# In[61]:


Personality_types = df1['type'].unique()
print(Personality_types)


# In[62]:


count = df1.groupby(['type']).count()
count


# In[63]:


def plot(data):
    plt.figure(figsize=(20,7))
    plt.xticks(fontsize=24, rotation=0)
    plt.yticks(fontsize=24, rotation=0)
    return sns.countplot(data=df1, x='type')


# In[64]:


plot(df1)


# In[65]:


count = df1.groupby(['type'])['posts'].count()
pie = go.Pie(labels=count.index, values=count.values)
figure = go.Figure(data=[pie])
py.iplot(figure)


# In[66]:


df1["NumPosts"] = df1["posts"].apply(lambda x: len(x.split("|||")))


# In[67]:


data


# In[68]:


df1


# In[69]:


s1=df1["NumPosts"].unique()
s1.sort()
print(s1)


# In[70]:


count=df1.groupby(['NumPosts']).count()
print(count[30:53])


# In[71]:


plt.figure(figsize=(35,10))
sns.displot(data=df1, x='NumPosts')


# In[72]:


df1[df1["NumPosts"]==4]


# In[73]:


def extract_urls(text):
  urls=re.findall(r'(https?:\/\/.*?[\s+])', text)
  return ",".join(urls)


# In[74]:


df1['url_links']=df1['posts'].apply(lambda x: extract_urls(x))


# In[75]:


df1


# In[76]:


def countyoutubelinks(posts):
    count = 0
    for p in posts:
        if 'youtube' in p:
            count += 1
    return count
        
df1['youtube_links'] = df1['url_links'].apply(lambda x: x.strip().split(",")).apply(countyoutubelinks)


# In[77]:


df1


# In[78]:


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


# In[79]:


len(df1[df1['I-E'] == 0])


# In[80]:


len(df1[df1['I-E'] == 1])


# In[81]:


len(df1[df1['N-S'] == 0])


# In[82]:


len(df1[df1['N-S'] == 1])


# In[83]:


len(df1[df1['T-F'] == 0])


# In[84]:


len(df1[df1['T-F'] == 1])


# In[85]:


len(df1[df1['J-P'] == 0])


# In[86]:


len(df1[df1['J-P'] == 1])


# In[87]:


personalities = df1.loc[: , "I-E":"J-P"].columns
for personality in personalities:
    sns.countplot(x=df1[personality],data=df1)
    plt.show()


# In[88]:


df_by_personality = df1.groupby("type")['posts'].apply(' '.join).reset_index()
df_by_personality


# In[89]:


def generate_wordcloud(text, title):
    wordcloud = WordCloud(background_color="white",width = 400 , height = 300).generate(text)
    plt.subplots(1 , 1 , figsize = [20,6])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize = 40)
    plt.show()


# In[90]:


for i, t in enumerate(df_by_personality['type']):
    text = df_by_personality.iloc[i,1]
    generate_wordcloud(text, t)


# # Pre Processing

# In[91]:


pd.set_option('display.max_colwidth',1000)


# In[92]:


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


# In[93]:


Stopwords = set(stopwords.words("english"))


# In[94]:


def stop_words(text):
  tweet_tokens = word_tokenize(text)
  filtered_words = [w for w in tweet_tokens if not w in Stopwords]
  return " ".join(filtered_words)


# In[95]:


def lemmantization(text):
  tokenized_text = word_tokenize(text)
  lemmatizer = WordNetLemmatizer()
  text = ' '.join([lemmatizer.lemmatize(a) for a in tokenized_text])
  return text


# In[96]:


def pre_process(text):
  text=replace_sep(text)
  text=remove_url(text)
  text=remove_punct(text)
  text=remove_numbers(text)
  text=convert_lower(text)
  text=extra(text)
  text=stop_words(text)
  text=lemmantization(text)
  return text


# In[97]:


data['cleaned_posts']= data['posts'].apply(lambda x: pre_process(x))


# In[ ]:


data.head(5)


# # Training the model

# In[ ]:


df=data


# In[ ]:


#Training Data
vector = CountVectorizer(analyzer="word", 
                             max_features=1500, 
                             tokenizer=None,    
                             preprocessor=None, 
                             stop_words=None,  
                             max_df=0.7,
                             min_df=0.1)
X=vector.fit_transform(df.cleaned_posts)
Y=np.array(df.type)


# In[ ]:


tfidf_transformer = TfidfTransformer()
X_final =tfidf_transformer.fit_transform(X) 


# In[ ]:


feature_names = list(enumerate(vector.get_feature_names()))
feature_names


# In[ ]:



print(Y.shape)
print(X.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X_final,Y,test_size=0.3,random_state=42)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_predictiontrain = random_forest.predict(X_train)
Y_predictiontest = random_forest.predict(X_test)

print("Train Accuracy:", np.mean(Y_predictiontrain == Y_train))
print("Test Accuracy:", np.mean(Y_predictiontest == Y_test))

accuracy_randomforest = (random_forest.score(X_train, Y_train) * 100)
print((accuracy_randomforest), "%")


# In[ ]:


from sklearn.svm import SVC
svm=SVC(kernel='linear')  
X_train,X_test,Y_train,Y_test=train_test_split(X_final,Y,test_size=0.3,random_state=42)
svm.fit(X_train,Y_train)

Y_predictiontrain = svm.predict(X_train)
Y_predictiontest = svm.predict(X_test)

print("Train Accuracy:", np.mean(Y_predictiontrain == Y_train))
print("Test Accuracy:", np.mean(Y_predictiontest == Y_test))


svm.score(X_train, Y_train)
acc_svm = round(svm.score(X_train, Y_train) * 100, 2)
print(round(acc_svm,2,), "%")


# # Testing

# In[ ]:


d3=data[8635:8675].copy()


# In[ ]:


d3.shape


# In[ ]:


def identify_personality(test):
  text=tfidf_transformer.transform(vector.transform([test])).toarray()
  all_words = vector.get_feature_names()
  t1 = pd.DataFrame.from_dict({words: text[:, i] for i, words in enumerate(all_words)})
  x = random_forest.predict(t1)
  return x


# In[ ]:


d3['predicted_personality']= d3['cleaned_posts'].apply(lambda x: identify_personality(x))


# In[ ]:


d3


# In[ ]:





# # Validation

# sample-1

# In[ ]:


data[8670:8671]


# In[ ]:


q1="ixfp always think cat fi doms reason especially website become neo nazi perc im nerd ive learning dutch duolingo im much fun duolingo shit oh god love xd right winger lack political consciousness doubt real hope hell theyre nothing like twilight vampire would agree likely related confidence level make sense someone would aggressive overcompensate sometimes perceived shortcoming nazi germany soviet union extremely nationalist cant think nationalist country nowadays dprk separate nationalism youre one plenty infps right wing conservativesalt reich fascist perc time zone patrick walker voice angel eh cloggies pretty cool people experience feel like going gym morning went anyway one best workout anyway dont feel like exercising anyway youll glad actually thought brand cologne first heard name laughing apparently tempbanned lot people forum theyre pretty dead late color ive come realize need spend time working improving skill learning new thing im usually lazy sitting around watching youtube playing video game dont felt like good time get fresh new name bit profile makeover george soros gave lot money kid contribute white genocide muslim arab move turn u caliphate im part honestly idea even since shahada gone see youre trying start personality cult replace right poly many tic blood sucking parasite youre welcome asking real question choice fuck either two men must choose one cant neither look exactly feel free use imagination please change name zeta neprok im tired old name thanks dont dissin bob ross eat pineapple pizza would leave embarrassment younger self thats youre smart fuck shahada always nice admit found sort confusing always odds crucial issue obviously shitpost thread stereotype ridiculous granted probably fit lot art skill horrendous always got ta love people use mbti stroke ego rolleyes im going say stupidity think people stupid anything like know many time nuclear war almost happened computer glitch somehow managed get coffee date weekend probably miracle sort im pretty drunk night hasnt even started yet laughing free market isnt free fact extremely coercive authoritarian reason state state institution influenced public sometimes working class people push reform anything feel comfortable around woman around men perc become neo nazi im pretty stupid whats interesting dont talk much lot people dont hear say stupid shit think im smart sucker laughing although disagree said multiculturalism assuming understood meant everything else said right made lot good point dont think he right say thats end goal multiculturalism would agree goal remove white dominance western society let honest thats course allowed sex white people otherwise youre sjw liberal cuck white child contributing white genocide also moderate even mean change depending context political situation time moderate would seen radical extremist people today would"


# In[ ]:


q2=tfidf_transformer.transform(vector.transform([q1])).toarray()


# In[ ]:


all_words = vector.get_feature_names()
q3 = pd.DataFrame.from_dict({words: q2[:, i] for i, words in enumerate(all_words)})


# In[ ]:


x=[]
x=random_forest.predict(q3)
y=x[0]


# In[ ]:


print(y)


# sample-2

# In[ ]:


data[8674:8675]


# In[ ]:


p1="long since personalitycafe although doesnt seem changed one bit must say good back somewhere like usually turn doctor overwhelmed world around one dream chased large shadowy creature someone else felt save else dream ended reached safety happened well avatar doctor clockwork creature always liked monster worker trying job kind st thanks reply appreciate help get nd think everyone right opinion however many people abuse right p yea iron man thing xd thanks advice everyone thanks think needed humour might show maybe know wont anything harsh like throw beat although place go really dont dont really know judge personality type say mum introverted dad extroverted p ok understand want feel liking men thats good problem tell parent want tell really dont know say im contempt always suppose spent afternoon buried book world spending morning animal suppose good mix whats hardest thing im going face life havent got clue current perspective telling parent im gay seems like huge hurdle suspect seem mi torchwood oh thats fictional oh well still got book two day ago read thought brilliant true family love never wanted deeply understand contempt keeping deeper self different people see look understand knowing important much know also happy let anything get lately feel better either guess wasnt watching comedy central incredible song thanks reply relate part least big deal may made reply pointing well something along line feel although ever entered state mind time need emotion sake sometimes think im young time live life find parent never happy effort give anything say used quite lazy lately tried best everything ask without hassle hey spoiler button still preload image doesnt matter im best driver currently hold restricted licence drive safely dont tell parent speed open road one around dont let someone tell belief wrong think right true people may agree make belief wrong mean midnight blue question many thing feel im contrasting may see dark shade blue fact story behind every line magnificent im fine ad make website know sometimes need reccomendation though make open new window clicked targettop yes want people understand people anyway think doesnt matter could understand many people dont care really fully interested know infjs idea letting people life keeping distance personaly people let world let one wouldnt bothered personally actually sometimes actually think dont realize obvious others might feel dont see good someone wont hang around happens chance wont notice accident good disappearing dont want see library walk somewhere remote alone agree although im black white person outside anyway website need colour bright black doesnt help im defiantly night person doubt enjoy late sleeping half day away say lost balance knew exactly feel say im similar place right lost thing really keep going balance world spinning look good think top navigation barnews would look better darker colour white seems bright would tenth way actually capt jack going say doctor thought second thing seemed appropriate apart tardis would get car drive stopping enjoy thanks solitude right like constructive criticism something work improve guess upset little feel people criticize work something whenever someone tell something done good enough make feel like good enough maybe yes owner lonely heart doctor fan say repeated memory wipe silence eventually fry brain eleventh doctor im remember people currently reading white wolf son good read quite expected picked finish soon next want read wolf gift tonight sit outside window blackness night staring wilderness call space peace beauty could stay till morning light thousand sun stare upon going close facebook month back well wanting able message family ausse school friend found connected website second mar collection seems fitting mood right seen agree actually think first time watched movie beginning got power kinda thought andrew would never work right ok watched underworld awakening must say really good film compared film last month anyway dont think good first would never want turn emotion sometimes hide world still need"


# In[ ]:


p2=tfidf_transformer.transform(vector.transform([p1])).toarray()


# In[ ]:


all_words = vector.get_feature_names()
p3 = pd.DataFrame.from_dict({words: p2[:, i] for i, words in enumerate(all_words)})


# In[ ]:


x=[]
x=random_forest.predict(p3)
y=x[0]


# In[ ]:


print(y)


# sample-3

# In[ ]:


data[8673:8674]


# In[ ]:


test="conflicted right come wanting child honestly maternal instinct whatsoever recently none close friend child guess closest friend isfj esfp istj xnfp esfj dont know correct dont know know type actually xnfp said last paragraph teacher frustrates there trend education combine class contain variety type student wont cant say much trouble community general mean im talking sameage peer never considered popular artsy extracurricular eat meat feel guilty vaguely considered eating meat dont think could actually nutritional requirement need high protein diet well boat except teach science theater know people dont think science infp find interesting im good another career exactly god truly created one man one woman technically created woman man product incest per book stayed list going around facebook choice following special order reader bernhard schlink interpreter malady jhumpa amazing list im partially posting im sure find need idea listen list isnt long others im pretty another point infp type known idealist heard theory procrastination say come person setting high standard mind ideal perhaps well isnt bit impulse control im feeling calm one else going feel calm cute dont think stereotype think eleanor roosevelt may infp google classified enfp read biography actually seems like introvert forced dont know much subtypes wow definitely teacher first time read thought yeah way reading carefully second time almost perfectly list interest hobby generally like read lot ill tell later though sometimes work get busy also enjoy writing come burst go well cant help heisnt gay thing sorry however commiserate part waiting someone text back feeling like power oh wow totally situation right im treated poorly opposite problem cant seem find courage stand ground dont think cry much mom think im pretty sure think im clinically depressed even though im mean im pretty sure im cry bit lol well feel dumb work see thanks alcohol doesnt actually change base personality easy change mbti wouldnt valid alcohol make say thing would absent site couple month since ive back couple week ago whenever try create thread like right reply thread unable create new also sorry huge long single paragraph reason window wont let create new paragraph attempting fix eggsies thanks reply perspective cheating thing interesting never really thought way especially beginning perhaps make good point infp forum conversation like getting gift agreed lot thing however im terrible gift giver anyone different taste im always good math never enjoyed math fair im slightly f opposed function also say math methodical ok mr enfj conundrum dating enfj since january busy schedule able spend little time love bike use one recreation actual transportation might bike weekend evening errand weather nice except right live steep hill id never make havent clinically diagnosed adhd reading extensively talking mom behavior kid found wanted take psychologist dad yoga clothes tight black capri pink purple tank tear movie sometimes dont generally cry hard exception movie version book read like harry potter believe two hour usually unfortunately kind read lot book fiction im working night circus erin morgenstern almost done dovekeepers alice hoffman also reading im little ambivalent book made movie many favorite book literary enough movie version would atrocious couple agree finish dark could read point without replying may physically alone someone hearing havent posted thread fear may belong curiosity someone desperately wanted meet need didnt know would tell person would even know ask well ive never read book seen looselybased movie see youre saying thought im fully convinced case im greatly appreciate input boxerkitty said enfjs need socialize wide set people time unwind make feel somewhat better one thing would love person choose spend life assuming find someone choose spend life similar point made completely agreed madrabbit exception im pursuing answer never find possibly doesnt exist go cycle get frustrated give decide stop yeah hear know really complicated wish seems way mind one dont entirely know bring without using word marriage dont know sure enfj seems far likely thought id post advice enfjs difficulty relationship something bugging week thought would post people seem get think thing way im seeing original post title anything go totally relate almost posted similar thread able trust relationship day ago desire control others kind freak feel like control feeling situation let explain desire tell someone else others mentioned feel really guilty spend lot money spatype thing feel like could putting somewhere useful especially since dont make ton sad angry pathetic embarrassed love oh definitely definitely stem combination number thing occurred life biggest overall factor form attachment incredibly easy seems im really hoping since referred breakup letter longdistance relationship seriously cant even face person tell feel cant honestly see point saying letter id like change cant imagine anyone said voted intp im closer e ok thing sort issue test took quiz got entertainer one dont think really type reason ill describe two yeah definitely get jealous come romantic relationship honest havent many serious romantic relationship part reason though certainly wait leafstone third volume q book originally split japanese another language spent entire week beach book didnt finish finished since"


# In[ ]:


text=tfidf_transformer.transform(vector.transform([test])).toarray()


# In[ ]:


all_words = vector.get_feature_names()
t1 = pd.DataFrame.from_dict({words: text[:, i] for i, words in enumerate(all_words)})


# In[ ]:


x=[]
x=random_forest.predict(t1)
y=x[0]


# In[ ]:


print(y)


# # Testing data using Twitter data

# In[ ]:


import tweepy as tw


# In[ ]:


consumer_key = 'DadKR3DKcG1PWvyh8igvAIaYN'
consumer_secret = 'KjWoOAwm7uwwT0vTGWcuomuPq9Wglo5pA29kPxhOPvddMmO2Eg'
access_token = '1266720191502680066-YGjG1jvAjIOOsG6NibYCDH7trAznfk'
access_token_secret = '1E8fi5w2hi8eRyvnVvHcJnOU9p66oiJaXe5l1PAIQNYqA'


# In[ ]:


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[ ]:


handle = '@sundarpichai'


# In[ ]:


res = api.user_timeline(screen_name=handle,count=50, include_rts=True)
tweets = [tweet.text for tweet in res]


# In[ ]:


tweets


# In[ ]:


def join(text):
  return "||| ".join(text)
   


# In[ ]:


t1=join(tweets)


# In[ ]:


t2=pre_process(t1)


# In[ ]:


t3=tfidf_transformer.transform(vector.transform([t2])).toarray()
all_words = vector.get_feature_names()
t4 = pd.DataFrame.from_dict({words: t3[:, i] for i, words in enumerate(all_words)})


# In[ ]:


x=random_forest.predict(t4)


# In[ ]:


print(x)


# In[ ]:


def twits(handle):
    res = api.user_timeline(screen_name=handle,count=50, include_rts=True)
    tweets = [tweet.text for tweet in res]
    return tweets


# In[ ]:


def twit(handle):
    res = api.user_timeline(screen_name=handle,count=50, include_rts=True)
    tweets = [tweet.text for tweet in res]
    t1=join(tweets)
    t2=pre_process(t1)
    t3=tfidf_transformer.transform(vector.transform([t2])).toarray()
    all_words = vector.get_feature_names()
    t4 = pd.DataFrame.from_dict({words: t3[:, i] for i, words in enumerate(all_words)})
    x=[]
    x=random_forest.predict(t4)
    
    y=x[0]
    return y


# In[ ]:


twit('@finkd')


# In[ ]:


a=twit('@BillGates')


# In[ ]:


print(a)


# In[ ]:


b=twit('@sundarpichai')


# In[ ]:


print(b)


# # Recommendations

# In[ ]:


def split(text):
  return [char for char in text]


# In[ ]:


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


# In[ ]:


List_ch_I=['Reflective',
           'Self-aware',
           'Take time making decisions',
           'Feel comfortable being alone',
           'Dont like group works']

List_ch_E=['Enjoy social settings',
           'Do not like or need a lot of alone time',
           'Thrive around people',
           'Outgoing and optimistic',
           'Prefer to talk out problem or questions']

List_ch_N=['Listen to and obey their inner voice',
           'Pay attention to their inner dreams',
           'Typically optimistic souls',
           'Strong sense of purpose',
           'Closely observe their surroundings']

List_ch_S=['Remember events as snapshots of what actually happened',
           'Solve problems by working through facts',
           'Programmatic',
           'Start with facts and then form a big picture',
           'Trust experience first and trust words and symbols less',
           'Sometimes pay so much attention to facts, either present or past, that miss new possibilities']

List_ch_F=['Decides with heart',
           'Dislikes conflict',
           'Passionate',
           'Driven by emotion',
           'Gentle',
           'Easily hurt',
           'Empathetic',
           'Caring of others']

List_ch_T=['Logical',
           'Objective',
           'Decides with head',
           'Wants truth',
           'Rational',
           'Impersonal',
           'Critical',
           'Firm with people']

List_ch_J=['Self-disciplined',
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

List_ch_P=['Relaxed',
           'Adaptable',
           'Nonjudgemental',
           'Carefree',
           'Creative',
           'Curious',
           'Postpone decisions to see what other options are available',
           'Act spontaneously',
           'Decide what to do as we do it, rather than forming a plan ahead of time',
           'Do things at the last minute']


# In[ ]:


def charcter(text):
    o=split(text)
    characteristics=[]
    for i in range(0,4):
        if o[i]=='I':
            characteristics.append('\n'.join(List_ch_I))
        if o[i]=='E':
            characteristics.append('\n'.join(List_ch_E))
        if o[i]=='N':
            characteristics.append('\n'.join(List_ch_N))
        if o[i]=='S':
            characteristics.append('\n'.join(List_ch_S))
        if o[i]=='F':
            characteristics.append('\n'.join(List_ch_F))
        if o[i]=='T':
            characteristics.append('\n'.join(List_ch_F))
        if o[i]=='J':
            characteristics.append('\n'.join(List_ch_J))
        if o[i]=='P':
            characteristics.append('\n'.join(List_ch_P))
    crct='\n'.join(characteristics)
    data = crct.split("\n")
    return data


# In[ ]:


def recomend(text):
    b=split(text)
    jobs=[]
    for i in range(0,4):
        if b[i]=='I':
            jobs.append('\n'.join(List_jobs_I))
        if b[i]=='E':
            jobs.append('\n'.join(List_jobs_E))
        if b[i]=='N':
            jobs.append('\n'.join(List_jobs_N))
        if b[i]=='S':
            jobs.append('\n'.join(List_jobs_S))
        if b[i]=='F':
            jobs.append('\n'.join(List_jobs_F))
        if b[i]=='T':
            jobs.append('\n'.join(List_jobs_T))
        if b[i]=='J':
            jobs.append('\n'.join(List_jobs_J))
        if b[i]=='P':
            jobs.append('\n'.join(List_jobs_P))
    crct1='\n'.join(jobs)
    data1=crct1.split("\n")
    return (split(data1))


# In[ ]:


def pp(handle):
    personality=twit(handle)
    return personality,recomend(personality),charcter(personality)


# In[ ]:


charcter(twit('@finkd'))


# In[ ]:


recomend(twit('@finkd'))


# # user_interface

# In[ ]:


from tkinter import *
from PIL import Image, ImageTk
import tkinter as tk
import os
class MyWindow:
    def __init__(self, win):
        self.bg1= ImageTk.PhotoImage(file="Home.png", master=window)
        canvas= Canvas( win,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        canvas.create_image(1250,70,image=self.bg1,anchor=NE)
        self.D_lbl0=Label(win, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(win, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=550,y=120)
        self.btn1 = Button(win, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.posts)
        self.btn1.place(x=750,y=120)
        
    def mbti(self):
        newwin=Toplevel(window)
        newwin.geometry("2600x2000+20+20")
        self.D_lbl0=Label(newwin, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin, text='MBTI DATA',bg='green',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin, text='MBTI TEST',bg='navy',fg='white',font=30,command=self.mbt)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin, text='EXPLORATORY DATA',bg='navy',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.D_lbl0=Label(newwin, text='The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides\n everyone into 16 distinct personality types across 4 axis:\n Introversion (I) — Extroversion (E)\nIntuition (N) — Sensing (S)\nThinking (T) — Feeling (F)\nJudging (J) — Perceiving (P)\nThe dataset contains 8675 observations (people), where each observation gives a person’s:\nMyers-Briggs personality type (as a 4-letter code)\nAn excerpt containing the last 50 posts on their PersonalityCafe forum (each entry separated by “|||”)\nFor example, someone who prefers introversion, intuition, thinking and perceiving would be\n labelled an INTP in the MBTI system, and there are lots of personality based components\n that would model or describe this person’s preferences or behaviour based on the label.\n',fg='black', font=("Helvetica", 25))
        self.D_lbl0.place(x=50,y=230)
        
    def mbt(self):
        newwin9=Toplevel(window)
        newwin9.geometry("2600x2000+20+20")
        self.D_lbl0=Label(newwin9, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin9, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin9, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin9, text='MBTI TEST',bg='green',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin9, text='EXPLORATORY DATA',bg='navy',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        canvas= Canvas( newwin9,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="TestResults.png")
        canvas.create_image(1150,70,image=self.bg1, anchor="ne")
        
    def explore(self):
        newwin3=Toplevel(window)
        newwin3.geometry("2600x2000+20+20")
        canvas= Canvas( newwin3,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="CountPlot.png")
        canvas.create_image(300,70,image=self.bg1, anchor="nw")
        
        self.D_lbl0=Label(newwin3, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin3, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin3, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin3, text='MBTI TEST',bg='navy',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin3, text='EXPLORATORY DATA',bg='green',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.btn1 = Button(newwin3, text='PIE PLOT',bg='navy',fg='white',font=30,command=self.explore1)
        self.btn1.place(x=150,y=250)
        self.btn1 = Button(newwin3, text='DIS PLOT',bg='navy',fg='white',font=30,command=self.explore2)
        self.btn1.place(x=150,y=320)
        self.btn1 = Button(newwin3, text='I-E PLOT',bg='navy',fg='white',font=30,command=self.explore3)
        self.btn1.place(x=150,y=390)
        self.btn1 = Button(newwin3, text='N-S PLOT',bg='navy',fg='white',font=30,command=self.explore4)
        self.btn1.place(x=150,y=460)
        self.btn1 = Button(newwin3, text='T-F PLOT',bg='navy',fg='white',font=30,command=self.explore5)
        self.btn1.place(x=150,y=530)
        self.btn1 = Button(newwin3, text='P-J PLOT',bg='navy',fg='white',font=30,command=self.explore6)
        self.btn1.place(x=150,y=600)
        
    def explore1(self):
        newwin6=Toplevel(window)
        newwin6.geometry("2600x2000+20+20")
        canvas= Canvas( newwin6,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="PiePlot.png")
        canvas.create_image(300,20,image=self.bg1, anchor="nw")
        
        
        self.D_lbl0=Label(newwin6, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin6, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin6, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin6, text='MBTI TEST',bg='navy',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin6, text='EXPLORATORY DATA',bg='green',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.btn1 = Button(newwin6, text='PIE PLOT',bg='green',fg='white',font=30,command=self.explore1)
        self.btn1.place(x=150,y=250)
        self.btn1 = Button(newwin6, text='DIS PLOT',bg='navy',fg='white',font=30,command=self.explore2)
        self.btn1.place(x=150,y=320)
        self.btn1 = Button(newwin6, text='I-E PLOT',bg='navy',fg='white',font=30,command=self.explore3)
        self.btn1.place(x=150,y=390)
        self.btn1 = Button(newwin6, text='N-S PLOT',bg='navy',fg='white',font=30,command=self.explore4)
        self.btn1.place(x=150,y=460)
        self.btn1 = Button(newwin6, text='T-F PLOT',bg='navy',fg='white',font=30,command=self.explore5)
        self.btn1.place(x=150,y=530)
        self.btn1 = Button(newwin6, text='P-J PLOT',bg='navy',fg='white',font=30,command=self.explore6)
        self.btn1.place(x=150,y=600)
        
    def explore2(self):
        newwin7=Toplevel(window)
        newwin7.geometry("2600x2000+20+20")
        canvas= Canvas( newwin7,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="Displot.png")
        canvas.create_image(300,20,image=self.bg1, anchor="nw")
        
        self.D_lbl0=Label(newwin7, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin7, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin7, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin7, text='MBTI TEST',bg='navy',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin7, text='EXPLORATORY DATA',bg='green',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.btn1 = Button(newwin7, text='PIE PLOT',bg='navy',fg='white',font=30,command=self.explore1)
        self.btn1.place(x=150,y=250)
        self.btn1 = Button(newwin7, text='DIS PLOT',bg='green',fg='white',font=30,command=self.explore2)
        self.btn1.place(x=150,y=320)
        self.btn1 = Button(newwin7, text='I-E PLOT',bg='navy',fg='white',font=30,command=self.explore3)
        self.btn1.place(x=150,y=390)
        self.btn1 = Button(newwin7, text='N-S PLOT',bg='navy',fg='white',font=30,command=self.explore4)
        self.btn1.place(x=150,y=460)
        self.btn1 = Button(newwin7, text='T-F PLOT',bg='navy',fg='white',font=30,command=self.explore5)
        self.btn1.place(x=150,y=530)
        self.btn1 = Button(newwin7, text='P-J PLOT',bg='navy',fg='white',font=30,command=self.explore6)
        self.btn1.place(x=150,y=600)
        
    def explore3(self):
        newwin8=Toplevel(window)
        newwin8.geometry("2600x2000+20+20")
        canvas= Canvas( newwin8,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="I_E.png")
        canvas.create_image(500,100,image=self.bg1, anchor="nw")
        
        self.D_lbl0=Label(newwin8, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin8, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin8, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin8, text='MBTI TEST',bg='navy',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin8, text='EXPLORATORY DATA',bg='green',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.btn1 = Button(newwin8, text='PIE PLOT',bg='navy',fg='white',font=30,command=self.explore1)
        self.btn1.place(x=150,y=250)
        self.btn1 = Button(newwin8, text='DIS PLOT',bg='navy',fg='white',font=30,command=self.explore2)
        self.btn1.place(x=150,y=320)
        self.btn1 = Button(newwin8, text='I-E PLOT',bg='green',fg='white',font=30,command=self.explore3)
        self.btn1.place(x=150,y=390)
        self.btn1 = Button(newwin8, text='N-S PLOT',bg='navy',fg='white',font=30,command=self.explore4)
        self.btn1.place(x=150,y=460)
        self.btn1 = Button(newwin8, text='T-F PLOT',bg='navy',fg='white',font=30,command=self.explore5)
        self.btn1.place(x=150,y=530)
        self.btn1 = Button(newwin8, text='P-J PLOT',bg='navy',fg='white',font=30,command=self.explore6)
        self.btn1.place(x=150,y=600)
        
    def explore4(self):
        newwin8=Toplevel(window)
        newwin8.geometry("2600x2000+20+20")
        canvas= Canvas( newwin8,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="N_S.png")
        canvas.create_image(500,100,image=self.bg1, anchor="nw")
        
        self.D_lbl0=Label(newwin8, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin8, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin8, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin8, text='MBTI TEST',bg='navy',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin8, text='EXPLORATORY DATA',bg='green',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.btn1 = Button(newwin8, text='PIE PLOT',bg='navy',fg='white',font=30,command=self.explore1)
        self.btn1.place(x=150,y=250)
        self.btn1 = Button(newwin8, text='DIS PLOT',bg='navy',fg='white',font=30,command=self.explore2)
        self.btn1.place(x=150,y=320)
        self.btn1 = Button(newwin8, text='I-E PLOT',bg='navy',fg='white',font=30,command=self.explore3)
        self.btn1.place(x=150,y=390)
        self.btn1 = Button(newwin8, text='N-S PLOT',bg='green',fg='white',font=30,command=self.explore4)
        self.btn1.place(x=150,y=460)
        self.btn1 = Button(newwin8, text='T-F PLOT',bg='navy',fg='white',font=30,command=self.explore5)
        self.btn1.place(x=150,y=530)
        self.btn1 = Button(newwin8, text='P-J PLOT',bg='navy',fg='white',font=30,command=self.explore6)
        self.btn1.place(x=150,y=600)  
        
    def explore5(self):
        newwin8=Toplevel(window)
        newwin8.geometry("2600x2000+20+20")
        canvas= Canvas( newwin8,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="T_F.png")
        canvas.create_image(500,100,image=self.bg1, anchor="nw")
        
        self.D_lbl0=Label(newwin8, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin8, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin8, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin8, text='MBTI TEST',bg='navy',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin8, text='EXPLORATORY DATA',bg='green',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.btn1 = Button(newwin8, text='PIE PLOT',bg='navy',fg='white',font=30,command=self.explore1)
        self.btn1.place(x=150,y=250)
        self.btn1 = Button(newwin8, text='DIS PLOT',bg='navy',fg='white',font=30,command=self.explore2)
        self.btn1.place(x=150,y=320)
        self.btn1 = Button(newwin8, text='I-E PLOT',bg='navy',fg='white',font=30,command=self.explore3)
        self.btn1.place(x=150,y=390)
        self.btn1 = Button(newwin8, text='N-S PLOT',bg='navy',fg='white',font=30,command=self.explore4)
        self.btn1.place(x=150,y=460)
        self.btn1 = Button(newwin8, text='T-F PLOT',bg='green',fg='white',font=30,command=self.explore5)
        self.btn1.place(x=150,y=530)
        self.btn1 = Button(newwin8, text='P-J PLOT',bg='navy',fg='white',font=30,command=self.explore6)
        self.btn1.place(x=150,y=600)
    def explore6(self):
        newwin8=Toplevel(window)
        newwin8.geometry("2600x2000+20+20")
        canvas= Canvas( newwin8,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        self.bg1= ImageTk.PhotoImage(file="J_P.png")
        canvas.create_image(500,100,image=self.bg1, anchor="nw")
        
        self.D_lbl0=Label(newwin8, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin8, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=590,y=120)
        self.btn1 = Button(newwin8, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=490,y=120)
        self.btn1 = Button(newwin8, text='MBTI TEST',bg='navy',fg='white',font=30)
        self.btn1.place(x=720,y=120)
        self.btn1 = Button(newwin8, text='EXPLORATORY DATA',bg='green',fg='white',font=30,command=self.explore)
        self.btn1.place(x=850,y=120)
        self.btn1 = Button(newwin8, text='PIE PLOT',bg='navy',fg='white',font=30,command=self.explore1)
        self.btn1.place(x=150,y=250)
        self.btn1 = Button(newwin8, text='DIS PLOT',bg='navy',fg='white',font=30,command=self.explore2)
        self.btn1.place(x=150,y=320)
        self.btn1 = Button(newwin8, text='I-E PLOT',bg='navy',fg='white',font=30,command=self.explore3)
        self.btn1.place(x=150,y=390)
        self.btn1 = Button(newwin8, text='N-S PLOT',bg='navy',fg='white',font=30,command=self.explore4)
        self.btn1.place(x=150,y=460)
        self.btn1 = Button(newwin8, text='T-F PLOT',bg='navy',fg='white',font=30,command=self.explore5)
        self.btn1.place(x=150,y=530)
        self.btn1 = Button(newwin8, text='P-J PLOT',bg='green',fg='white',font=30,command=self.explore6)
        self.btn1.place(x=150,y=600)
    
    def twitter(self):
        newwin4=Toplevel(window)
        newwin4.geometry("2600x2000+20+20")
        self.D_lbl0=Label(newwin4, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin4, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=350,y=120)
       # self.btn1 = Button(newwin4, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        #self.btn1.place(x=750,y=120)
        self.btn1 = Button(newwin4, text='TWITTER POSTS',bg='navy',fg='white',font=30,command=self.posts)
        self.btn1.place(x=480,y=120)
        self.btn1 = Button(newwin4, text='PREDICTED PERSONALITY',bg='navy',fg='white',font=30,command=self.home)
        self.btn1.place(x=640,y=120)
        self.btn1 = Button(newwin4, text='RECOMENDATIONS',bg='navy',fg='white',font=30,command=self.recomends)
        self.btn1.place(x=870,y=120)
        
    def posts(self):
        newwin2=Toplevel(window)
        newwin2.geometry("2600x2000+20+20")
        self.D_btn1 = Button(newwin2, text='TWITTER POSTS',bg='green',fg='white',font=30,command=self.posts)
        self.D_btn1.place(x=480,y=120)
        self.D_b1=Button(newwin2, text='PREDICT PERSONALITY',bg='navy',fg='white',font=30,command=self.home)
        self.D_b1.place(x=670,y=120)
        self.D_btn1 = Button(newwin2, text='RECOMENDATIONS',bg='navy',fg='white',font=30,command=self.recomends)
        self.D_btn1.place(x=930,y=120)
        self.btn1 = Button(newwin2, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=350,y=120)
        #self.btn1 = Button(newwin2, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        #self.btn1.place(x=750,y=120)
        self.D_lbl0=Label(newwin2, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.t1=Text(newwin2)
        self.t3=Text(newwin2)
        self.t2=Entry(newwin2,font=150,width=30)
        self.lbl1=Label(newwin2, text='Enter Twitter ID: ',bg='navy',fg='white', font=("Helvetica", 30))
        self.lbl1.place(x=60,y=230)
        self.lbl4=Label(newwin2,text='Tweets data of user:',bg='navy',fg='white',font=("Helvetica",15))
        self.lbl4.place(x=150,y=350)
        self.lbl4=Label(newwin2,text='Cleaned data:',bg='navy',fg='white',font=("Helvetica",15))
        self.lbl4.place(x=850,y=350)
        self.t1.place(x=150,y=380)
        self.t3.place(x=850,y=380)
        self.t2.place(x=600,y=230,height=45)
        self.b1=Button(newwin2, text='Get_Tweets',bg='green',fg='white',font=70,command=self.twt)
        self.b1.place(x=400, y=290,width=130,height=50) 
        self.b1=Button(newwin2, text='PreProcess Tweets',bg='green',fg='white',font=70,command=self.twt1)
        self.b1.place(x=800, y=290,width=170,height=50) 
        
    def twt(self):
        handle=self.t2.get()
        res=twits(handle)
        self.t1.insert(END,str(res))
        
    def twt1(self):
        handle=self.t2.get()
        res1=twits(handle)
        tx1=join(res1)
        tx2=pre_process(tx1)
        self.t3.insert(END,str(tx2))
        
    def recomends(self):
        newwin1=Toplevel(window)
        newwin1.geometry("2600x2000+20+20")
        self.D_btn1 = Button(newwin1, text='TWITTER POSTS',bg='navy',fg='white',font=30,command=self.posts)
        self.D_btn1.place(x=480,y=120)
        self.D_b1=Button(newwin1, text='PREDICT PERSONALITY',bg='navy',fg='white',font=30,command=self.home)
        self.D_b1.place(x=670,y=120)
        self.D_btn1 = Button(newwin1, text='RECOMENDATIONS',bg='green',fg='white',font=30,command=self.recomends)
        self.D_btn1.place(x=930,y=120)
        self.btn1 = Button(newwin1, text='HOME',fg='white',bg='navy',font=30,command=self.home1)
        self.btn1.place(x=350,y=120)
       # self.btn1 = Button(newwin1, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        #self.btn1.place(x=750,y=120)
        self.D_lbl0=Label(newwin1, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.lbl1=Label(newwin1, text='Enter handle name: ',bg='navy',fg='white',font=("Helvetica",25))
        self.lbl2=Label(newwin1, text='Job_Recommendations:',bg='navy',fg='white',font=("Helvetica",15))
        #self.lbl4=Label(text='Charecterstics of a person:',bg='navy',fg='white',font=("Helvetica",15))
        self.lbl5=Label(newwin1,text='Personality Type:',bg='navy',fg='white',font=("Helvetica",15))
        self.b1=Button(newwin1, text='Recomendations',bg='green',fg='white',font=40, command=self.recmd)
        self.b1.place(x=700, y=290)
        self.t0=Entry(newwin1,font=100)
        self.t2=Text(newwin1,height=15,width=85)
        self.t1=Entry(newwin1,font=100)
        self.t0.place(x=700,y=220,height=40)
        self.lbl2.place(x=400,y=410)
        self.lbl1.place(x=400,y=220)
        self.lbl5.place(x=500,y=330)
        self.t1.place(x=680,y=330)
        self.t2.place(x=400,y=460) 
        
    def recmd(self):
        handle=self.t0.get()
        res=twit(handle)
        self.t1.insert(END,str(res))
        r=self.t1.get()
        result=recomend(res)
        for i in range(len(result)):
            self.t2.insert(END, str(result[i]))
            self.t2.insert(END, str('\n'))   
     
    def home(self):
        newwin5=Toplevel(window)
        newwin5.geometry("2600x2000+20+20")
        self.D_btn1 = Button(newwin5, text='TWITTER POSTS',bg='navy',fg='white',font=30,command=self.posts)
        self.D_btn1.place(x=480,y=120)
        self.D_b1=Button(newwin5, text='PREDICT PERSONALITY',bg='green',fg='white',font=30,command=self.home)
        self.D_b1.place(x=670,y=120)
        self.D_btn1 = Button(newwin5, text='RECOMENDATIONS',bg='navy',fg='white',font=30,command=self.recomends)
        self.D_btn1.place(x=930,y=120)
        self.btn1 = Button(newwin5, text='HOME',bg='navy',fg='white',font=30,command=self.home1)
        self.btn1.place(x=350,y=120)
        #self.btn1 = Button(newwin5, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.twitter)
        #self.btn1.place(x=750,y=120)
        self.D_lbl0=Label(newwin5, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.lbl2=Label(newwin5, text='Characteristics of Personalities:',bg='navy',fg='white' ,font=("Helvetica", 25))
        self.lbl2.place(x=830,y=250)
        self.t=Text(newwin5,height=15,width=85)
        self.t.place(x=830,y=310)
        self.lbl2=Label(newwin5, text='Predicted Personality Type ',bg='navy',fg='white' ,font=("Helvetica", 25))
        self.lbl2.place(x=30,y=430)
        self.lbl1=Label( newwin5,text='Enter handle name of twitter: ',bg='navy',fg='white', font=("Helvetica", 25))
        self.lbl1.place(x=30,y=300)
        self.t1=Entry(newwin5,bd=3,font=100)
        self.t1.place(x=480,y=300,height=40)
        self.t2=Entry(newwin5,bd=3,font=100)
        self.t2.place(x=480,y=430,height=40)
        self.b1=Button(newwin5, text='Predict_personality',bg='green',fg='white',font=70, command=self.predict)
        self.b1.place(x=400, y=380)  
        
    def predict(self):
        handle=self.t1.get()
        res=twit(handle)
        self.t2.insert(END,str(res))
        r=self.t2.get()
        result=charcter(res)
        for i in range(len(result)):
            self.t.insert(END, str(result[i]))
            self.t.insert(END, str('\n'))  
    def home1(self):
        newwin15=Toplevel(window)
        newwin15.geometry("2600x2000+20+20")
        self.bg1= ImageTk.PhotoImage(file="Home.png")
        canvas= Canvas( newwin15,width= 2500, height= 2000)
        canvas.pack(expand=True, fill= BOTH)
        canvas.pack(padx=0,pady=170)
        canvas.create_image(1250,70,image=self.bg1, anchor="ne")
        self.D_lbl0=Label(newwin15, text='Personality Based Job Recommender Using Twitter Data ',fg='navy', font=("Helvetica", 40))
        self.D_lbl0.place(x=110,y=30)
        self.btn1 = Button(newwin15, text='MBTI DATA',bg='navy',fg='white',font=30,command=self.mbti)
        self.btn1.place(x=550,y=120)
        self.btn1 = Button(newwin15, text='TWITTER DATA',bg='navy',fg='white',font=30,command=self.posts)
        self.btn1.place(x=750,y=120)
window=tk.Tk()
mywin=MyWindow(window)
window.geometry("2600x2000+20+20")
window.mainloop()


# In[ ]:




