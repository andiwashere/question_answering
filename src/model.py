from transformers import TFAlbertModel
import tensorflow as tf
import numpy as np
import src.utils as u


class QaAlbertModel(tf.keras.Model):
    """Pretrained Albert Model with dense layer and QA head.
       Mostly following https://huggingface.co/transformers/model_doc/bert.html#tfbertforquestionanswering.
       Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **start_scores**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads."""
    def __init__(self, config, *inputs, **kwargs):
        super(QaAlbertModel, self).__init__(config, *inputs, **kwargs)

        # config needs to be a dict
        self.linear_hidden_size = 128
        self.num_labels = config['num_labels']

        # set inputs to None (sequence length not known) as this is a condition for being able to save the model
        self.sequence = tf.keras.Input(shape=(None,), dtype=np.int32)

        self.albert = TFAlbertModel.from_pretrained('huggingface/')

        # compute score of each output of a word t_i being the start or end of the answer:
        # s*t_i, e*t_j (s: start vector, e: end vector) in top layer with num_labels=2
        self.qa_linear = tf.keras.layers.Dense(
            self.linear_hidden_size, kernel_initializer=u.get_initializer(config['initializer_range']), name="qa_linear"
        )
        self.qa_outputs = tf.keras.layers.Dense(
            config['num_labels'], kernel_initializer=u.get_initializer(config['initializer_range']), name="qa_outputs"
        )
        # sum over sequence
        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.albert.trainable = False

        # has to come at the end of this constructor, otherwise other layers are unknown
        self._set_inputs(self.sequence)

    def call(self, inputs, **kwargs):
        outputs = self.albert(inputs, **kwargs)

        sequence_output = outputs[0]

        hidden_output = self.qa_linear(sequence_output)
        logits = self.qa_outputs(hidden_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        start_output = self.softmax(start_logits)
        end_output = self.softmax(end_logits)

        # add hidden states and attention if they are here
        outputs = (tf.stack([start_output, end_output], axis=1),) + outputs[2:]

        return outputs  # start_logits, end_logits, (hidden_states), (attentions)

    @property
    def get_linear_hidden_size(self):
        return self.linear_hidden_size