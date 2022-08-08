# Prerequistes
import nltk
from ast import keyword
from email import header
from attr import has
import pandas as pd
import numpy as np
from requests import head
import streamlit as st
import warnings
import joblib
import re

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title='Restaurant Finder', layout='wide')

if 'btn' not in st.session_state:
    st.session_state['btn']=False

if 'state' not in st.session_state:
    st.session_state['state']= 'All'

if 'city' not in st.session_state:
    st.session_state['city']= 'All'

if st.session_state['state']=='All':
    st.session_state['city']='All'

if 'cuisine' not in st.session_state:
    st.session_state['cuisine']='All'

if 'veg' not in st.session_state:
    st.session_state['veg']='All'

if 'n_result' not in st.session_state:
    st.session_state['n_result']=20

if 'sort' not in st.session_state:
    st.session_state['sort']='None'


# filter warnings
warnings.filterwarnings('ignore')

# Function to load restaurants data, similarities and restaurant keywords
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_data():
    texts= joblib.load('texts.sav')
    similarities= joblib.load('similarities.sav')
    df= pd.read_csv('cleaned_data.csv')
    df.drop(['highlights','cuisine_count','hdi','hdi_category'], axis=1,inplace=True)
    df.cuisines= df.cuisines.apply(lambda x: re.sub("[\[\]']",'',x)).str.strip(' ').str.split(',')
    df.high= df.high.apply(lambda x: re.sub("[\[\]']",'',x)).str.strip(' ').str.split(',')
    similarities= pd.DataFrame(similarities, index= df.index,columns= df.index)
    return df, texts, similarities

# Function for asserting if a keyword or multiple keywords exists in a text.
def has_all_keys(x, values):
    splited_text= values.split()
    keywords= x.split(' ')
    res= []
    for k in keywords:
        res.append(k in splited_text)
    return all(res)

# Function for removing stopwords from search inputs
def remove_stopwords(text):
    tokens= [word.lower().strip() for word in text.split(' ') if word not in stopwords.words('english')]
    tokens= list(dict.fromkeys(tokens))
    return ' '.join(tokens)

# Function for getting similar restaurant recommendations
def recommendations(ids, city):
    try:
        filters= sim[ids[0]].sort_values(ascending=False)[1:16].index
        try:
            filters.append(sim[ids[1]].sort_values(ascending=False)[1:16].index)
            filters.append(sim[ids[2]].sort_values(ascending=False)[1:16].index)
            filters.append(sim[ids[3]].sort_values(ascending=False)[1:16].index)
        except:
            pass
        filters= list(set(filters))
        custom_dict= dict(zip(filters,range(len(filters))))
        if city is 'All':
            return df[(df.index.isin(filters))].sort_index(key= lambda x: x.map(custom_dict))
        else:
            return df[(df.index.isin(filters)) & (df.city==city)].sort_index(key= lambda x: x.map(custom_dict))
    except:
        pass

# Function for finding restaurants based on search
def search_for_restaurant(keyword, state, city, cuisines, veg):
    if city is 'All':
        pass
    else:
        keyword= keyword+' '+city
    if cuisines is 'All':
        pass
    else:
        keyword= keyword+ ' '+ cuisines
    keyword= remove_stopwords(keyword)
    filters= texts[texts.apply(lambda x: has_all_keys(keyword, x))].index.tolist()
    custom_dict= dict(zip(filters,range(len(filters))))
    filtered_data= df[df.index.isin(filters)].sort_index(key= lambda x: x.map(custom_dict))
    if state is 'All':
        pass
    else:
        filtered_data= filtered_data[filtered_data.state==state]
    if veg is 'All':
        pass
    else:
        if veg=='Yes':
            filtered_data= filtered_data[filtered_data.veg==1]
        else:
            filtered_data= filtered_data[filtered_data.veg==0]
    return filtered_data

# Function for showing search results and recommendations
def show_results(results):
    res_iter= results.itertuples()
    nrows= int(np.ceil(len(results)/3))
    for _ in range(nrows):
        with st.container():
            contcols= st.columns(3)
        for contcol in contcols:
            try:
                row= next(res_iter)
                with contcol.expander(label= row.name+' | '+row.city):
                    st.text(row.address+', '+row.state)
                    st.text(' |'.join(row.cuisines))
                    st.text(' ●'.join(row.high))
                    st.text(str(row.aggregate_rating)+ '★')
                    st.text(str(row.votes)+' votes')
                    st.text('₹'+str(row.average_cost_for_two)+ ' for two')
                    st.markdown("[ Open on Zomato ]({})".format(row.url), unsafe_allow_html=True)
            except:
                contcol.empty()


def main():
    with open('./style.css') as css:
        html= "<style>{}</style>".format(css.read())
    
    st.markdown(html,unsafe_allow_html=True)
    st.header('Search Restaurants, Cuisines or Locations.')
    headercols= st.columns(2)
    searchbox= headercols[0].text_input('', key= 'searchbox')
    btn= headercols[1].button('🔍')

    if btn:
        st.session_state['btn']= True
    
    if st.session_state['btn']:
        if st.session_state['searchbox']=='':
            st.write('Please type something and try again.')
        else:
            cols= st.columns(9)
            st.session_state['state']= cols[0].selectbox('State', options= all_dict.keys())
            st.session_state['city']= cols[1].selectbox('City', options= all_dict.get(st.session_state['state']))
            st.session_state['cuisine']= cols[2].selectbox('Cuisines',options= cuisines_list)
            st.session_state['veg']= cols[3].radio('Pure Vegetarian', options= ['All','Yes','No'])
            st.session_state['sort']= cols[-2].selectbox('Sort by', ['None','Highly Rated', 'Price Low to High', 'Price High to Low'])
            st.session_state['n_result']= cols[-1].slider('Show N records', min_value=10, max_value=50)
            for col in cols[5:-3]: col.empty()

            st.subheader("Showing results for '{}'".format(st.session_state['searchbox']))

            search_results= search_for_restaurant(keyword= st.session_state['searchbox'], city= st.session_state['city'], 
                                                  cuisines=st.session_state['cuisine'], state= st.session_state['state'],
                                                  veg= st.session_state['veg'])
            
            if search_results.shape[0]!=0:
                display_res= search_results
                if st.session_state['sort']=='Price Low to High':
                    display_res= display_res.sort_values(by=['average_cost_for_two','votes'], ascending=[True,False])
                elif st.session_state['sort']=='Price High to Low':
                    display_res= display_res.sort_values(by=['average_cost_for_two','votes'], ascending=False)
                elif st.session_state['sort']=='Highly Rated':
                    # display_res= display_res[display_res.aggregate_rating>=4.0]
                    display_res= display_res.sort_values(by=['aggregate_rating','votes'], ascending=False)
                elif st.session_state['sort']=='None':
                    display_res= display_res.sort_values(by='votes', ascending=False)
                show_results(display_res.iloc[:st.session_state['n_result'],:])
            else:
                st.write("Sorry, we can't find anything.")
            
            recom= recommendations(search_results.index, city= st.session_state['city'])
            if recom is not None:
                st.markdown('Similar restaurants you can try')
                show_results(recom.iloc[:20,:])
                st.header("That's all we've got!")
    

if __name__=='__main__':
    
    # Load data
    df, texts, sim= load_data()
    
    cities_dict= cities_dict= df.groupby('state').city.apply(lambda x: ['All']+ list(x.unique())).to_dict()
    all_dict= {'All': ['All']}
    all_dict.update(cities_dict)
    
    cuisines_list= df.cuisines.explode().str.strip().value_counts()[:15].index.tolist()
    cuisines_list.insert(0, 'All')
    
    main()





