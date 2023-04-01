import tensorflow as tf

class TopicAttentiveEmbedding(tf.keras.layers.Layer):
    # Layer for calculating the beta/topic-attentive representation (check TopicExpan paper) of the topic and document embedding
    # aka. softmax of the bilinear interaction of the topic embedding with the embedding of each word in the document
    # needs to use the same trained weights as the bilinear layer in the Similarity Predictor
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def call(self, topic_embedding: tf.Tensor, sequence_embedding: tf.Tensor, shared_bilinear_layer: tf.keras.layers.Layer):
        # topic_embedding shape = (batch_size, topic_embedding_len)
        # sequence_embedding shape = (batch_size, max_len, 768)

        reshaped_topic_embedding: tf.Tensor = tf.reshape(topic_embedding, (-1, 1, topic_embedding.shape[1]))
        reshaped_topic_embedding: tf.Tensor = tf.repeat(reshaped_topic_embedding, sequence_embedding.shape[1], axis=1)

        def apply_bilinear(embeddings: tuple[tf.Tensor, tf.Tensor]):
            emb1, emb2 = embeddings
            bilinear_embedding = tf.reshape(shared_bilinear_layer([emb1, emb2], training=False), shape=[-1])
            softmax_bilinear_embedding = tf.nn.softmax(bilinear_embedding)
            return softmax_bilinear_embedding
        
        beta_embedding = tf.map_fn(
            apply_bilinear, 
            elems=(reshaped_topic_embedding, sequence_embedding),
            fn_output_signature=tf.TensorSpec(shape=(sequence_embedding.shape[1]))
        )
        
        return beta_embedding