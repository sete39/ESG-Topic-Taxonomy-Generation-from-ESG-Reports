import pickle
import pandas as pd
from dataclasses import dataclass
from bertopic._bertopic import BERTopic
from numpy.typing import NDArray
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic 
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print('Reading dataset...')
with open('../dataset/docs.pkl', 'rb') as fp: 
    docs, docs_info = pickle.load(fp)

with open('../dataset/esg_classification.pkl', 'rb') as handle:
    results = pickle.load(handle)

result_df = pd.DataFrame(results)
docs_df = pd.DataFrame({'docs': docs})
info_df = pd.DataFrame({'page_number': [page_number for page_number, _ in docs_info], 'report_url': [report_url for _, report_url in docs_info]})

print(len(docs_df))

df = pd.concat([docs_df, result_df, info_df], axis=1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True).loc[:,~df.columns.duplicated()].copy() # shuffle

del docs, results, result_df, docs_df

df.loc[(df['score'] < 0.7) & (df['label'] == 'Governance'), 'label'] = 'None'
df.loc[(df['score'] < 0.7) & (df['label'] == 'Social'), 'label'] = 'None' 
df.loc[(df['score'] < 0.7) & (df['label'] == 'Environmental'), 'label'] = 'None' 

environmental_topic_list = [['Climate change', 'Biofuels', 'Climate change strategy', 'Emissions management', 'Emissions reporting'],
                   ['Ecosystem service', 'Land access', 'Biodiversity management', 'Water'],
                   ['Pollution control', 'Waste', 'Recycling']]

social_topic_list = [['Public health', 'Access to medicine', 'Nutrition', 'Product safety'],
                   ['Human rights', 'Community relations', 'Privacy free expression', 'Security'],
                   ['Labor standards', 'Diversity', 'Health', 'Safety', 'Supply chain labor standards'],
                   ['Society', 'Charity', 'Education', 'Employment']]

governance_topic_list = [['Corporate governance', 'Audit', 'Control', 'Board structure', 'Remuneration', 'Shareholder rights', 'Transparency', 'Talent'],
                   ['Business ethics', 'Bribery', 'Corruption', 'Political influence'],
                   ['Stakeholder engagement']]    


@dataclass
class ChildTopicModel:
    bert_topic_model: BERTopic
    child_docs_indices: List[int]
        
@dataclass
class RootTopicModel:
    topic_model: BERTopic
    docs: list
    embeddings: NDArray
    child_topic_models: dict
        

# second-level topic modeling
print('Finding second-level topics...')
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

main_representation = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)]

esg_topics = [
    ('Environmental', 8, environmental_topic_list), 
    ('Social', 8, social_topic_list), 
    ('Governance', 5, governance_topic_list)
]


esg_topic_models = {}
for root_topic, num_topics, seed_topic_list in esg_topics:
    cluster_model = MiniBatchKMeans(n_clusters=num_topics)
    docs = df[df['label']==root_topic]['docs'].to_list()
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1,3), min_df=5, max_features=10000)

    topic_model = BERTopic(
                        representation_model=main_representation,
                        vectorizer_model=vectorizer_model,
                        embedding_model=sentence_model,
                        hdbscan_model=cluster_model,
                        calculate_probabilities=False,
                        seed_topic_list=seed_topic_list,
                        verbose=True)
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    esg_topic_models[root_topic] = RootTopicModel(topic_model, np.copy(docs), np.copy(embeddings), [])


# Union-Find implementation from https://stackoverflow.com/questions/20154368/union-find-implementation-using-python
def union_find(lis):
    lis = map(set, lis)
    unions = []
    for item in lis:
        temp = []
        for s in unions:
            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        unions = temp
    return unions

def get_topic_embeddings(topic_model, n, model_name='all-MiniLM-L6-v2'):
    # generating topic embeddings based on weighted avg. of the top-n terms in a topic
    topic_embeddings = []
    range_for_loop = range(0, len(topic_model.get_topic_info()) - 1) if topic_model.get_topic_info()['Topic'].iloc[0] == -1 else range(len(topic_model.get_topic_info()))
    phrase_sentence_model = SentenceTransformer(model_name)

    for lower_topic_number in range_for_loop:
        top_n_words = [term_tuple[0] for term_tuple in topic_model.get_topic(lower_topic_number)[:n]]
        word_embeddings = phrase_sentence_model.encode(top_n_words)

        word_importance = [term_tuple[1] for term_tuple in topic_model.get_topic(lower_topic_number)[:n]]
        topic_embedding = np.average(word_embeddings, weights=word_importance, axis=0)
        topic_embeddings.append(topic_embedding)

    return topic_embeddings

def find_parent_outliers(topic_model, topic_model_lower, filtered_docs, n: int, topic_number):
    # finds wether the subtopic should be assigned to another topic, and marks them as outlier if so
    # getting most relevant parent topic of each subtopic
    topic_embeddings = get_topic_embeddings(topic_model_lower, 5)
    topic_embeddings.insert(0, np.zeros(len(topic_embeddings[0])))
    
    sim_matrix = cosine_similarity(topic_embeddings, topic_model.topic_embeddings_)
    assigned_parent_topic_indices = np.argmax(sim_matrix, axis=1)
    
    # marking topics as outliers
    lower_topics_df = topic_model_lower.get_topic_info()
    lower_topics_df['parent_topic'] = [topic_model.get_topic_info()['Name'].iloc[index] for index in assigned_parent_topic_indices]
    lower_topics_df['parent_topic_number'] = assigned_parent_topic_indices

    topics_to_merge = [[-1, row['Topic']] for i, row in lower_topics_df.iterrows() if row['parent_topic_number'] != topic_number]
    
    topic_model_lower.merge_topics(filtered_docs, topics_to_merge)

def merge_redundant_topics(topic_model_lower, filtered_docs, n: int):
    topic_embeddings = get_topic_embeddings(topic_model_lower, n)

    sim_matrix_covariance = cosine_similarity(topic_embeddings, topic_embeddings)
    np.fill_diagonal(sim_matrix_covariance, 0)
    rows, cols = np.where(sim_matrix_covariance > 0.80)
    to_merge = [(row, col) for row, col in zip(rows, cols)]
    
    if len(to_merge) == 0:
        return
    
    to_merge = [(row, col) for row, col in zip(rows, cols)]
    merge_sets_list = [sorted(merge_set) for merge_set in union_find(to_merge)]
    to_merge = [(list(merge_set)[0], second_node) for merge_set in merge_sets_list for second_node in list(merge_set)[1:]]
    
    topic_model_lower.merge_topics(filtered_docs, to_merge)
b
    
def reduce_model_outliers(topic_model_lower, filtered_docs, filtered_embeddings):
    # minimizes number of outlier documents
    new_topics = topic_model_lower.reduce_outliers(filtered_docs, topic_model_lower.topics_, strategy="embeddings", embeddings=filtered_embeddings, threshold=0.3)
    new_topics = np.array(new_topics)
    if len(new_topics[new_topics == -1]) > 0:
        new_topics = topic_model_lower.reduce_outliers(filtered_docs, new_topics , strategy="c-tf-idf", threshold=0.3)
    new_topics = np.array(new_topics)
    
    documents = pd.DataFrame({"Document": filtered_docs, "Topic": new_topics})
    topic_model_lower._update_topic_size(documents)
    
def find_subtopics(topic_number, docs_df, docs, embeddings, lower_vectorizer_model, topic_model):
    # finds third-level topics for a second-level topic model
    filtered_indices = list(docs_df[docs_df['Topic'] == topic_number].index)
    filtered_docs = np.array(docs)[filtered_indices]
    filtered_embeddings = embeddings[filtered_indices]
    lower_lvl_seed = np.array([term_tuple[0] for term_tuple in topic_model.get_topic(topic_number)[:5]]).reshape(-1, 1).tolist()
        
    min_topic_size = 25
    if len(filtered_docs) > 20000:
        min_topic_size = 50
    elif len(filtered_docs) > 30000:
        min_topic_size = 75
    elif len(filtered_docs) > 40000:
        min_topic_size = 100
        
    topic_model_lower = BERTopic(
        representation_model=[KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)],
        vectorizer_model=lower_vectorizer_model,
        embedding_model=sentence_model,
        calculate_probabilities=False,
        seed_topic_list=lower_lvl_seed,
        min_topic_size=min_topic_size,
        verbose=True)
    
    topics_lower, probs = topic_model_lower.fit_transform(filtered_docs, filtered_embeddings)
    
    find_parent_outliers(topic_model, topic_model_lower, filtered_docs, 3, topic_number)
    reduce_model_outliers(topic_model_lower, filtered_docs, filtered_embeddings)
    merge_redundant_topics(topic_model_lower, filtered_docs, 5)

    return ChildTopicModel(topic_model_lower, filtered_indices)

print('Finding third-level topics...')
for root_topic, root_topic_model in esg_topic_models.items():
    child_topic_models_dict = {}
    for topic_number in list(root_topic_model.topic_model.get_topics())[0:]:
        lower_vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=1, max_features=10000)
        docs_df = root_topic_model.topic_model.get_document_info(root_topic_model.docs)
        
        child_topic_model = find_subtopics(topic_number, docs_df, root_topic_model.docs, root_topic_model.embeddings, lower_vectorizer_model, root_topic_model.topic_model)
        child_topic_models_dict[topic_number] = child_topic_model
    esg_topic_models[root_topic].child_topic_models = child_topic_models_dict
    

# Creating visualization and taxonomy results JSON
terms_list = [(0, 'root')]
i = 1
root_taxonomy_dict = {}

# converts an n-level (1, 2, or 3) key into a topic
# i.e. (0, 9, 18) corresponds to the the first top level topic,
# its 9th child, and 18th child ofSA the 9th child
key_to_topic = {} 
split_base_terms = [root_topic_model.topic_model.get_topic(base_topic_number)[0][0] for root_topic, root_topic_model in esg_topic_models.items() for base_topic_number in list(root_topic_model.child_topic_models.keys())]
    
for root_topic_number, (root_topic, root_topic_model) in enumerate(esg_topic_models.items()):
    terms_taxonomy_dict = {}
    root_topic_term = (i, root_topic)
    terms_list.append(root_topic_term)
    key_to_topic[(root_topic_number)] = root_topic_term
    
    i += 1
    for j, base_topic_number in enumerate(list(root_topic_model.child_topic_models.keys())):
        child_topic_models_dict = root_topic_model.child_topic_models
        child_topic_model = child_topic_models_dict[base_topic_number].bert_topic_model
        
        topics_dict = child_topic_model.get_topics()
        base_topic = (i, root_topic_model.topic_model.get_topic(base_topic_number)[0][0])
        
        key_to_topic[(root_topic_number, j)] = (root_topic_term, base_topic)
        terms_list.append(base_topic)
        
        child_term_list = []
        for k, (child_topic_number, terms) in enumerate(topics_dict.items()):
            i += 1
            split_terms = [term_tuple[1] for term_tuple in terms_list]
            split_child_terms = [term_tuple[1] for term_tuple in child_term_list]
            child_topic = (i, next(term for term, probability in terms if (term != base_topic[1]) and (term not in split_terms) and (term not in split_child_terms) and (term not in split_base_terms)))
            child_term_list.append(child_topic)
            
            key_to_topic[(root_topic_number, j, k)] = (root_topic_term, base_topic, child_topic)
            
        terms_list = terms_list + child_term_list
        terms_taxonomy_dict[base_topic] = child_term_list
        i += 1
        
    root_taxonomy_dict[root_topic_term] = terms_taxonomy_dict

    
def assign_to_topics(topic_model, docs, embeddings, n: int, method='positive_means', doc_indices=None, n_stds=1, model_name='all-MiniLM-L6-v2', has_outliers=False):
    # Uses transform to get the single topic assigning from BERTopic
    # Then measuring similarity of doc to topic to find other similar topics
    assigned_docs, _ = topic_model.transform(docs, embeddings=embeddings)
    assigned_docs = np.array(assigned_docs)
    phrase_sentence_model = SentenceTransformer(model_name)

    if doc_indices is None:
        doc_indices = list(range(len(docs)))
    doc_indices = np.array(doc_indices)
    
    range_for_loop = range(0, len(topic_model.get_topic_info()) - 1) if topic_model.get_topic_info()['Topic'].iloc[0] == -1 else range(len(topic_model.get_topic_info()))
    
    topic_embeddings = []
    for lower_topic_number in range_for_loop:
        top_n_words = ', '.join([term_tuple[0] for term_tuple in topic_model.get_topic(lower_topic_number)[:n]])
        topic_embeddings.append(phrase_sentence_model.encode(top_n_words))
    docs_to_topic_sims = cosine_similarity(embeddings, topic_embeddings)
    
    
    model_topics = topic_model.get_topics()
    if -1 in model_topics:
        del model_topics[-1]
        
    range_for_loop = range(-1, len(model_topics)) if has_outliers else range(len(model_topics))
    topic_assigned_indices = [(assigned_docs == i).nonzero()[0] for i in range_for_loop]
    
    if method == 'positive_means':
        topic_means = [docs_to_topic_sims[topic_assigned_indices[i]].mean(axis=0)[i] - docs_to_topic_sims[topic_assigned_indices[i]].var(axis=0)[i] for i in range_for_loop]
    else:
        topic_means = docs_to_topic_sims.mean(axis=0) + (n_stds * docs_to_topic_sims.std(axis=0))
    
    topic_to_docs = {i: doc_indices[(docs_to_topic_sims[:, i] > topic_means[i]).nonzero()[0]] for i in range_for_loop}
    for i, assigned_topic_number in enumerate(assigned_docs):
        doc_number = doc_indices[i]
        topic_to_docs[assigned_topic_number] = np.unique(np.append(topic_to_docs[assigned_topic_number], doc_number))
    
    doc_to_topics = {doc_number: [] for doc_number in doc_indices}
    
    for topic_number, docs_list in topic_to_docs.items():
        [doc_to_topics[doc_number].append(topic_number) for doc_number in docs_list]
        
    return topic_to_docs, doc_to_topics


print('Assigning second-level topics to documents...')
upper_topic_docs_list = []
second_level_topic_docs_list = []
for esg_topic_name in esg_topic_models:
    topic_model = esg_topic_models[esg_topic_name].topic_model
    docs = esg_topic_models[esg_topic_name].docs
    embeddings = esg_topic_models[esg_topic_name].embeddings
    upper_topic_to_docs, upper_doc_to_topics = assign_to_topics(topic_model, docs, embeddings, 5, method='all_means', n_stds=1.5)
    
    upper_topic_docs_list.append((upper_topic_to_docs, upper_doc_to_topics))
    
    topic_docs_list = []
    for topic_number, doc_indices in list(upper_topic_to_docs.items()):
        filtered_docs = np.array(docs)[doc_indices]
        filtered_embeddings = embeddings[doc_indices]
        topic_model_lower = esg_topic_models[esg_topic_name].child_topic_models[topic_number].bert_topic_model
        
        topic_to_docs, doc_to_topics = assign_to_topics(topic_model_lower, filtered_docs, filtered_embeddings, 5, method='all_means', doc_indices=doc_indices, n_stds=1, has_outliers=True)
        topic_docs_list.append((topic_to_docs, doc_to_topics))
    
    second_level_topic_docs_list.append(topic_docs_list)
    print('Finished ' + esg_topic_name)
    
    
print('Assigning third-level topics to documents...')
from copy import deepcopy
doc_to_topics_dict = {}

for i, upper_topic_docs in enumerate(upper_topic_docs_list):
    esg_key = list(esg_topic_models.keys())[i]
    empty_list = [[] for _ in range(len(esg_topic_models[esg_key].docs))]
    doc_to_topics_dict[esg_key] = { 'l2_topics': empty_list, 'l3_topics': deepcopy(empty_list) }
    
    for doc_number, second_level_topic_numbers in upper_topic_docs[1].items():
        for topic_number in second_level_topic_numbers:
            third_level_topics = second_level_topic_docs_list[i][topic_number][1][doc_number]
            doc_to_topics_dict[esg_key]['l2_topics'][doc_number].append(key_to_topic[(i, topic_number)])
            [doc_to_topics_dict[esg_key]['l3_topics'][doc_number].append(key_to_topic[(i, topic_number, third_level_topic_number)]) for third_level_topic_number in third_level_topics if third_level_topic_number != -1]
            

print('Creating dataframe with assigned topics...')
for esg_key in esg_topic_models.keys():
    l2_topics = [[both_topics[1] for both_topics in doc_topics] for doc_topics in doc_to_topics_dict[esg_key]['l2_topics']]
    l3_topics = [[both_topics[2] for both_topics in doc_topics] for doc_topics in doc_to_topics_dict[esg_key]['l3_topics']]
    
    df.loc[(df['label'] == esg_key), 'l2_topics'] = l2_topics
    df.loc[(df['label'] == esg_key), 'l3_topics'] = l3_topics
    

# Tree plotting for visualization
# Plotting code bsaed on https://plotly.com/python/tree-plots/
from sklearn import preprocessing
import igraph as ig
from igraph import Graph, EdgeSeq

n_vertices = len(terms_list)
edges = []

for root_topic, terms_taxonomy_dict in root_taxonomy_dict.items():
    encoded_root = root_topic[0]
    edges.append([0, encoded_root])
    
    for base_topic, child_topic_list in terms_taxonomy_dict.items():
        encoded_base = base_topic[0]
        encoded_children = [child_topic[0] for child_topic in child_topic_list] 

        edges.append([encoded_root, encoded_base])
        new_edges = [[encoded_base, child] for child in encoded_children]
        edges = edges + new_edges

G = ig.Graph(n_vertices, edges)
G.vs['name'] = terms_list
layt = G.layout_reingold_tilford(mode="all", root=[0])
labels = terms_list

from plotly.graph_objs import *
import plotly

N=n_vertices
E=[e.tuple for e in G.es]# list of edges

Xn=[layt[k][0] for k in range(N)]
Yn=[layt[k][1] for k in range(N)]
Xe=[]
Ye=[]
for e in E:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    
trace1 = Scatter(x=Xe,
               y=Ye,
               mode='lines',
               line= dict(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )
trace2 = Scatter(x=Xn,
   y=Yn,
   mode='markers',
   name='ntw',
   marker=dict(symbol='circle-dot',
        size=5,
        color='#6959CD',
        line=dict(color='rgb(50,50,50)', width=0.5)
        ),
   text=labels,
   hoverinfo='text'
   )

axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )
width=1200
height=800
layout=Layout(
    title='ESG - Environmental Topic Taxonomy',
    font= dict(size=10),
    showlegend=False,
    autosize=False,
    width=width,
    height=height,
    xaxis=layout.XAxis(axis),
    yaxis=layout.YAxis(axis),
    margin=layout.Margin(
        l=40,
        r=40,
        b=85,
        t=100,
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            xref='paper',
            yref='paper',
            x=0,
            y=-0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=12
            )
            )
        ]
    )

data=[trace1, trace2]
fig=Figure(data=data, layout=layout)
fig.update_yaxes(autorange='reversed')

plotly.offline.plot(fig, filename='./complete_taxonomy_visualization.html')

print('Visualization created at ./complete_taxonomy_visualization.html')

# creating taxonomy and docs jsons
def find_all_docs_with_topic(df: pd.DataFrame, l2_topic: List = None, l3_topic: List = None):
    # gets all elements that match a topic and returns a df with only that topic
    if not l2_topic and not l3_topic:
        print('Add atleast one topic')
        return
    
    if l3_topic:
        df_no_na = df.dropna(subset='l3_topics')
        filtered_df = df_no_na[df_no_na['l3_topics'].apply(lambda x: l3_topic in x)]
    else:
        df_no_na = df.dropna(subset='l2_topics')
        filtered_df = df_no_na[df_no_na['l2_topics'].apply(lambda x: l2_topic in x)]
        filtered_df = filtered_df[filtered_df['l3_topics'].apply(lambda x: x == [])]
    
    return filtered_df

def convert_row_to_dict(row):
    return {
        "url": row['report_url'],
        "text": row['docs'],
        "page": row['page_number']
    }

restructured_taxonomy_dict = {}
for esg_topic, esg_taxonomy_dict in root_taxonomy_dict.items():
    str_esg_topic = str(esg_topic[0]) + ' ' + esg_topic[1]
    
    restructured_second_level_dict = {}
    for second_level_topic, second_level_taxonomy in esg_taxonomy_dict.items():
        restructured_second_level_dict[str(second_level_topic[0]) + ' ' + second_level_topic[1]] = {str(topic_number) + ' ' + topic_name: {'docs': find_all_docs_with_topic(df, l3_topic=(topic_number, topic_name)).index.to_list()} for topic_number, topic_name in second_level_taxonomy}
        
        l2_topics_dict_list = find_all_docs_with_topic(df, l2_topic=second_level_topic).index.to_list()
        restructured_second_level_dict[str(second_level_topic[0]) + ' ' + second_level_topic[1]]['docs'] = l2_topics_dict_list
        
    restructured_taxonomy_dict[str_esg_topic] = restructured_second_level_dict
    
df_no_na = df.dropna(subset='l2_topics')
x = df_no_na.apply(convert_row_to_dict, axis=1).to_list()

import json
with open("../dataset/topic_taxonomy_docs.json", "w") as outfile:
    json.dump(x, outfile)
    
with open("../dataset/topic_taxonomy.json", "w") as outfile:
    json.dump(restructured_taxonomy_dict, outfile)