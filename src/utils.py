import tensorflow as tf
import numpy as np
import json
from itertools import compress
from transformers import AlbertTokenizer
from data.squad.squad_preprocess import preprocess

def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
       Taken from https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFPreTrainedModel.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    Description on tensorflow api:
        These values are similar to values from a random_normal_initializer except that values more
        than two standard deviations from the mean are discarded and re-drawn.
        This is the recommended initializer for neural network weights and filters.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def get_hms_string(s):
    m = s // 60
    s %= 60
    h = m // 60
    m %= 60
    return "{}h:{}min:{:.1f}s".format(int(h), int(m), s)


def get_dataset(data_path, pretrained_path):
    """
    Further preprocessing steps to make train/dev squad dataset usable as a tf.data.Dataset
    :param data_path: where .json file with data is located
    :param pretrained_path: where model and tokenizer of a huggingface transformer model is located
    :return: dataset: a tf.data.Dataset containing ((input_tokens, token_type_ids), labels)
    :return: input_tokens: tokenized version of question + context
    """
    contexts, questions, answers_start, answers_text, titles = preprocess(data_path)

    # convert character based answers_start to being based on tokens (25th char -> 3rd token)
    tokenizer = AlbertTokenizer.from_pretrained(pretrained_path)
    answers_start_char = [len(tokenizer.encode(questions[i], contexts[i][:start])) - 1 for i, start in
                          enumerate(answers_start)]

    # calculate answers_end from answers_start and answers_text, subtract 3 instead of 1 because of start and separating token of tokenizer
    answers_end_char = [
        answers_start_char[i] + len(tokenizer.encode(answers_text[i])) - 3 if answers_text[i] is not None else None for
        i in range(len(questions))]

    # tokenize questions and contexts together for correct placement of separating tokens
    input_tokens, input_token_type_ids = [], []
    for i, q in enumerate(questions):
        tokens_output = tokenizer.encode_plus(q, contexts[i], return_attention_mask=False, return_token_type_ids=True)
        input_tokens.append(tokens_output['input_ids'])
        input_token_type_ids.append(tokens_output['token_type_ids'])

    # as input into ALBERT is restricted to a length 512, get a mask
    with open(pretrained_path + 'config.json') as f:
        config = json.load(f)
    seq_len = config['max_position_embeddings']
    input_tokens_len = np.array([len(inp) for inp in input_tokens])
    mask_length = np.array([input_tokens_len <= seq_len]).reshape(len(questions))
    # filter questions with no answers
    mask_no_ans = np.array([True if a is not None else False for a in answers_end_char]).reshape(len(questions))
    mask = [all(tup) for tup in zip(mask_length, mask_no_ans)]

    # concatenate answers_start and answers_end
    answers = [[start, end] for start, end in zip(answers_start_char, answers_end_char)]
    answers = list(compress(answers, mask))
    input_tokens = list(compress(input_tokens, mask))
    input_token_type_ids = list(compress(input_token_type_ids, mask))

    # convert all lists to ragged tensors and then to dataset
    answers_tf = tf.ragged.constant(answers)
    input_tokens_tf = tf.ragged.constant(input_tokens)
    input_token_type_ids_tf = tf.ragged.constant(input_token_type_ids)
    answers_data = tf.data.Dataset.from_tensor_slices(answers_tf)
    input_tokens_data = tf.data.Dataset.from_tensor_slices(input_tokens_tf)
    input_token_type_ids_data = tf.data.Dataset.from_tensor_slices(input_token_type_ids_tf)
    in_data = tf.data.Dataset.zip((input_tokens_data, input_token_type_ids_data))
    dataset = tf.data.Dataset.zip((in_data, answers_data))

    return dataset, input_tokens