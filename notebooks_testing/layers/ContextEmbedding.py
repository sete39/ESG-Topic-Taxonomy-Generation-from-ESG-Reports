import tensorflow as tf

class ContextEmbedding(tf.keras.layers.Layer):
    # Layer for calculating the context (Q) representation (check TopicExpan paper) of the topic and document embedding
    # aka. the beta representation (which combines the topic and document representations) multiplied by the respective
    # word in the document representation
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]):
        topic_attentive_embedding, sequence_embedding = inputs
        context_embedding = tf.multiply(sequence_embedding, topic_attentive_embedding)
        return context_embedding
