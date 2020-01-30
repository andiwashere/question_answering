import tensorflow as tf
import numpy as np
import json, time, datetime, pytz, os
from src.utils import get_hms_string
from itertools import compress
from transformers import AlbertTokenizer
from src.model import QaAlbertModel
from data.squad.squad_preprocess import preprocess
"""
Training script for fine-tuning Google's ALBERT model with custom head on the SQUAD Dataset.
As my GPU can only pass a batch size of 2 through QaAlbertModel, accumulate gradients.
To start using this script, download the ALBERT model from huggingface and save a pretrained
tf model and tokenizer in your PRETRAINED_PATH. Furthermore, get the dataset from the squad website
and update SQUAD_TRAIN_DATA_PATH
"""
EPOCHS = 200
LR = [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7]
BETA1 = 0.9
BETA2 = 0.99
BATCH_SIZE = 2
DESIRED_BATCH_SIZE = 32 # should be multiple of BATCH_SIZE
USE = 0.005             # how much of dataset to use
CHECK_AVG_LOSS = 0.1    # check that loss doesn't vanish/explode after this much of an epoch
GRADS_BOUNDS = [1e-10, 1e+10]
SAVE_MODEL = True
SHUFFLE = False
SHUFFLE_BUF = 1000      # buffer size for dataset shuffling
TENSORBOARD = True
LOAD_DATA = True

# paths
PRETRAINED_PATH = 'huggingface/'
SQUAD_TRAIN_DATA_PATH = 'data/squad/train-v2.0.json'
MODEL_SAVE_PATH = 'models/with_answers_only/'
TRAIN_LOG_DIR = 'logs/gradient_tape/'

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if TENSORBOARD:
    %load_ext tensorboard
    %tensorboard --logdir logs/gradient_tape

if LOAD_DATA:
    contexts, questions, answers_start, answers_text, titles = preprocess(SQUAD_TRAIN_DATA_PATH)

    # convert character based answers_start to being based on tokens (25th char -> 3rd token)
    tokenizer = AlbertTokenizer.from_pretrained(PRETRAINED_PATH)
    answers_start_char = [len(tokenizer.encode(questions[i], contexts[i][:start])) - 1 for i, start in enumerate(answers_start)]

    # calculate answers_end from answers_start and answers_text, subtract 3 instead of 1 because of start and separating token of tokenizer
    answers_end_char = [answers_start_char[i] + len(tokenizer.encode(answers_text[i])) - 3 if answers_text[i] is not None else None for i in range(len(questions))]

    # tokenize questions and contexts together for correct placement of separating tokens
    input_tokens, input_token_type_ids = [], []
    for i, q in enumerate(questions):
        tokens_output = tokenizer.encode_plus(q, contexts[i], return_attention_mask=False, return_token_type_ids=True)
        input_tokens.append(tokens_output['input_ids'])
        input_token_type_ids.append(tokens_output['token_type_ids'])

    # as input into ALBERT is restricted to a length 512, get a mask
    with open(PRETRAINED_PATH + 'config.json') as f:
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
    train_dataset = tf.data.Dataset.zip((in_data, answers_data)).batch(BATCH_SIZE)
    if SHUFFLE:
        train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUF, reshuffle_each_iteration=True)

else:
    tokenizer = AlbertTokenizer.from_pretrained(PRETRAINED_PATH)
    with open(PRETRAINED_PATH + 'config.json') as f:
        config = json.load(f)

for lr in LR:
    model = QaAlbertModel(config)
    linear_hidden_size = model.get_linear_hidden_size
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=BETA1, beta_2=BETA2)
    model.compile(loss=loss_fn, optimizer=optimizer)

    print(f"SETUP DONE - START TRAINING with lr: {lr}")
    train_loss_results = []
    train_accuracy_results = []
    weird_grads = False
    current_time = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d-%H%M%S")
    model_weights_path = MODEL_SAVE_PATH + 'lr{}_'.format(lr) + current_time
    train_log_dir = TRAIN_LOG_DIR + 'lr{}_'.format(lr) + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    """ 
    TODO:   - add validation set
            - what about questions with no answers? 
            - for evaluation, compute score of all answer spans with s*t_i + e*t_j (s: start vector, e: end vector)
    """
    for e in range(EPOCHS):
        epoch_start = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_grads_avg = tf.keras.metrics.Mean()

        tape = tf.GradientTape()
        batch_times, i, grads_idx = [], 0, 0
        accumulated_grads_0 = tf.zeros((config['hidden_size'], linear_hidden_size))
        accumulated_grads_1 = tf.zeros([linear_hidden_size])
        accumulated_grads_2 = tf.zeros((linear_hidden_size, config['num_labels']))
        accumulated_grads_3 = tf.zeros((config['num_labels']))
        len_train_data = int(USE * np.ceil(len(input_tokens)/DESIRED_BATCH_SIZE))
        take_data = int(USE * np.ceil(len(input_tokens) / BATCH_SIZE))

        for inputs, labels in train_dataset.take(take_data):
            if grads_idx % DESIRED_BATCH_SIZE is 0:
                # time and index for desired batch
                i += 1
                t = time.time()
            grads_idx += BATCH_SIZE

            # due to difficulties with ragged tensors, pad inputs and transform them to regular tensor
            inputs_pad = inputs[0].to_tensor(default_value=0)
            token_type_ids = inputs[1].to_tensor(default_value=1)
            labels = labels.to_tensor()

            # Optimize the model
            with tape:
                preds = model(inputs=inputs_pad, token_type_ids=token_type_ids, training=True)[0]
                loss = loss_fn(y_pred=preds, y_true=labels)

                grads = tape.gradient(loss, model.trainable_variables)
                accumulated_grads_0 += grads[0]
                accumulated_grads_1 += grads[1]
                accumulated_grads_2 += grads[2]
                accumulated_grads_3 += grads[3]

                if grads_idx % DESIRED_BATCH_SIZE is 0 or int(grads_idx/BATCH_SIZE) is take_data:
                    optimizer.apply_gradients(zip([accumulated_grads_0, accumulated_grads_1, accumulated_grads_2, accumulated_grads_3], model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss)
            epoch_accuracy(labels, preds)
            epoch_grads_avg(tf.reduce_mean(tf.math.abs(accumulated_grads_0)) + tf.reduce_mean(tf.math.abs(accumulated_grads_2)))

            if grads_idx % DESIRED_BATCH_SIZE is 0 or int(grads_idx/BATCH_SIZE) is take_data:
                # check that avg grads don't vanish/explode
                if i % int(CHECK_AVG_LOSS * len_train_data) is 0:
                    grads_mean = epoch_grads_avg.result()
                    if grads_mean < GRADS_BOUNDS[0] or grads_mean > GRADS_BOUNDS[1]:
                        weird_grads = True
                        with train_summary_writer.as_default():
                            abort_statement = f'Due to grads being {grads_mean}, this configuration will be aborted at {current_time}.\n' + \
                                              f'LR: {lr} - EPOCH: {e} - ITER: {i} - BETAS: {BETA1}, {BETA2}\n'
                            print(abort_statement)
                            tf.summary.text('ABORT TRAINING', abort_statement, step=1.0)
                        break

                batch_times.append(time.time() - t)
                total_time_left = np.mean(batch_times) * (len_train_data - i - 1 + (EPOCHS - e - 1) * len_train_data)
                print("Epoch {}: Step {} / {} took {:.3f}s (~{} left) - Train batch loss: {}".format(
                    e, i, len_train_data, batch_times[-1], get_hms_string(total_time_left), loss), end='\r')

                # set accumulated values to 0 and reset tape for next gradient update
                with tape:
                    tape.reset()
                accumulated_grads_0 = tf.zeros((config['hidden_size'], linear_hidden_size))
                accumulated_grads_1 = tf.zeros([linear_hidden_size])
                accumulated_grads_2 = tf.zeros((linear_hidden_size, config['num_labels']))
                accumulated_grads_3 = tf.zeros((config['num_labels']))

        # End epoch
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=e)
            tf.summary.scalar('accuracy', epoch_accuracy.result(), step=e)
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        print(f"Epoch {e} took {get_hms_string(time.time()-epoch_start)} and finished with: "
              f"avg loss: {epoch_loss_avg.result()} - avg accuracy: {epoch_accuracy.result()}")
        epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()
        epoch_grads_avg.reset_states()
        if weird_grads is True:
            break

        if not os.path.exists(model_weights_path):
            os.makedirs(model_weights_path)
        model.save_weights(model_weights_path + '/model', save_format='tf')

