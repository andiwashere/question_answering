import tensorflow as tf


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
