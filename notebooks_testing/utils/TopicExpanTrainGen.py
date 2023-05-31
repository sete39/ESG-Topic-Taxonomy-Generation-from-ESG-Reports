import math
from spektral.data.utils import to_disjoint
from random import shuffle
import numpy as np
import tensorflow as tf

def shift_padded_array_right(arr: np.ndarray):
    # replaces last non-zero with a zero (for teacher-forcing)
    # return arr
    def apply_right_padding(arr: np.ndarray):
        # print(arr)
        first_zero_index = (arr==0).argmax(axis=0)
        shifted = arr.copy()
        shifted[first_zero_index-1] = 0
        return shifted
    
    return np.apply_along_axis(
        apply_right_padding,
        axis=1,
        arr=arr
    )

def shift_padded_array_left(arr: np.ndarray, max_len: int):
    # (for teacher-forcing)
    shifted_arr = np.roll(arr, -1)
    shifted_arr[: :, max_len-1] = 0
    return shifted_arr


class TopicExpanTrainGen(tf.keras.utils.Sequence):
    # Creating a custom Keras generator for model.fit()
    # Splits batches into mini-batches of size (mini_batch_size)
    # batch size must be a multiple of mini_batch_size

    # TODO: Shuffle the dataset on epoch end
    def __init__(self, topic_graph_list: list, document_input: np.array, document_topics: np.array, document_terms: list, batch_size: int, mini_batch_size: int, encoded_topic_to_tokenized_dict: dict):
        if batch_size % mini_batch_size != 0:
            raise Exception('batch_size must be a multiple of mini_batch_size')
        if len(document_input) != len(document_topics):
            raise Exception('Document list must be equal to the label list in length')
        
        self.topic_graph_list = topic_graph_list
        self.document_input = document_input
        self.document_topics = document_topics
        self.document_terms = document_terms
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.encoded_topic_to_tokenized_dict = encoded_topic_to_tokenized_dict

        # used for creating batches (to get random documents for the negative samples of the batch)
        # Stores a list of indexes which match the given topic (key)
        self.document_topics_dict = {}
        for i, topic in enumerate(document_topics):
            if topic not in self.document_topics_dict:
                self.document_topics_dict[topic] = [i]
            else:
                self.document_topics_dict[topic].append(i)

        # ego-graph list
        self.graph_features = []
        # maps a topic to its ego-graph
        self._topic_to_graph_dict = {}

        for graph in topic_graph_list:
            self.graph_features.append(graph)
            
            label = graph.y[2] if graph.y[2] is not None else graph.y[1] if graph.y[1] is not None else graph.y[0]

            self._topic_to_graph_dict[label] = graph

    def __len__(self):
        return math.ceil(len(self.document_topics) / (self.batch_size / self.mini_batch_size))

    def __getitem__(self, non_batch_index: int):
        # retrieving batch at non_batch_index
        batch_to_mini_batch_ratio = int((self.batch_size / self.mini_batch_size))
        idx = non_batch_index * batch_to_mini_batch_ratio
        
        max_len = len(self.document_input[0])
        
        batch_graph_list = []
        batch_document_list = []
        batch_similarity_label = []
        batch_phrase_list = []
        for mini_batch in range(batch_to_mini_batch_ratio):
            # creating mini_batch / getting the positive document
            positive_doc_idx = min(idx + mini_batch, len(self.document_topics) - 1)
            positive_doc_topic = self.document_topics[positive_doc_idx]
            positive_document = [self.document_input[positive_doc_idx]]
            positive_graph = self._topic_to_graph_dict[positive_doc_topic]

            exclude_list = [positive_doc_topic]
            negative_documents = []
            mini_batch_phrase_list = [self.document_terms[positive_doc_idx]] 
            for _ in range(self.mini_batch_size-1):
                # creating mini_batch / getting random negative documents for the rest of the minibatch
                random_negative_topic = np.random.choice(list(set([x for x in self.document_topics_dict.keys()]) - set(exclude_list)))
                exclude_list.append(random_negative_topic)

                random_negative_document_idx = np.random.choice(self.document_topics_dict[random_negative_topic])
                random_negative_document = self.document_input[random_negative_document_idx]
                negative_documents.append(random_negative_document)

                mini_batch_phrase_list.append(self.document_terms[random_negative_document_idx])
            
            # label for the similarirt prediction is 1 for the positive document (first element),
            # and zero for all negative documents
            mini_batch_similarity_label = [1] + ([0] * (self.mini_batch_size-1))

            # shuffling mini batch
            to_shuffle_list = list(zip(mini_batch_similarity_label, (positive_document + negative_documents), mini_batch_phrase_list))
            shuffle(to_shuffle_list)
            mini_batch_similarity_label, mini_batch_document_list, mini_batch_phrase_list = zip(*to_shuffle_list)
            # mini_batch_document_list = (positive_document + negative_documents)
            batch_graph_list.append(np.array([positive_graph for _ in range(self.mini_batch_size)]))
            batch_document_list.append(mini_batch_document_list)
            batch_phrase_list.append(mini_batch_phrase_list)
            batch_similarity_label.append(mini_batch_similarity_label)
        
        # GNN batched inputs
        batch_graph_list = np.array(batch_graph_list).flatten()
        x_in, a_in, i_in = to_disjoint(
            x_list=[g.x for g in batch_graph_list],
            a_list=[g.a for g in batch_graph_list]
        )
        a_in_sparse_tensor = tf.sparse.SparseTensor(
            indices=np.array([a_in.row, a_in.col]).T,
            values=a_in.data,
            dense_shape=a_in.shape
        )
        # document encoder batch inputs
        batch_document_list = np.array(batch_document_list).reshape((-1, max_len))
        
        # similarity predictor batch labels
        batch_similarity_label = np.array(batch_similarity_label).reshape((-1, 1))
        
        # phrase generator phrases/labels
        batch_phrase_list = np.array(batch_phrase_list).reshape((-1, max_len))

        # combining inputs/features and labels into tuples 
        model_inputs = (
            tf.convert_to_tensor(x_in, dtype=tf.float32), 
            a_in_sparse_tensor, 
            tf.convert_to_tensor(i_in, dtype=tf.int32), 
            tf.convert_to_tensor(batch_document_list),
            tf.convert_to_tensor(shift_padded_array_right(batch_phrase_list).reshape((-1, max_len)))
        )
        model_outputs = (
            tf.convert_to_tensor(batch_similarity_label), 
            tf.convert_to_tensor(shift_padded_array_left(batch_phrase_list, max_len))
        )
        return model_inputs, model_outputs

    # def on_epoch_end(self):
    #     # shuffling mini batch
    #     to_shuffle_list = list(zip(self.document_input, self.document_topics))
    #     shuffle(to_shuffle_list)
    #     self.document_input, self.document_topics = zip(*to_shuffle_list)
    #     return super().on_epoch_end()