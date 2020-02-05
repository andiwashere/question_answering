import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import json, time, datetime, pytz, os, sys, itertools
from src.utils import get_hms_string, get_dataset
from transformers import AlbertTokenizer
from src.model import QaAlbertModel
"""
Training script for fine-tuning Google's ALBERT model with custom head on the SQUAD Dataset.
As my GPU can only pass a batch size of 2 through QaAlbertModel, accumulate gradients.
To start using this script, download the ALBERT model from huggingface and save a pretrained
tf model and tokenizer in your PRETRAINED_PATH. Furthermore, get the dataset from the squad website
and update SQUAD_TRAIN_DATA_PATH
"""
# hyperparameters
EPOCHS = 35
LR = [1e-3]
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.0              # L2 regularization
DROPOUT = False
BATCH_SIZE = 2                  # needs to be 2 for my GPU
DESIRED_BATCH_SIZES = [16]  # should be multiple of BATCH_SIZE

# data and visualization
USE = 0.2                       # how much of dataset to use
CHECK_AVG_LOSS = 1              # check that loss doesn't vanish/explode after this much of an epoch
GRADS_BOUNDS = [1e-6, 1e+6]
SAVE_MODEL = True
SHUFFLE = False
SHUFFLE_BUF = 1000              # buffer size for dataset shuffling
TENSORBOARD = True
LOAD_DATA = True

# paths
PRETRAINED_PATH = 'huggingface/'
SQUAD_TRAIN_DATA_PATH = 'data/squad/train-v2.0.json'
SQUAD_VAL_DATA_PATH = 'data/squad/dev-v2.0.json'
MODEL_SAVE_PATH = 'models/with_answers_only/'
LOG_DIR = 'logs/gradient_tape/'

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if TENSORBOARD:
    %load_ext tensorboard
    %tensorboard --logdir logs/gradient_tape

if LOAD_DATA:
    train_dataset, input_tokens_train = get_dataset(SQUAD_TRAIN_DATA_PATH, PRETRAINED_PATH)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset, input_tokens_val = get_dataset(SQUAD_VAL_DATA_PATH, PRETRAINED_PATH)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    if SHUFFLE:
        # doesn't work with reshuffle_each_iteration --> other data at every epoch?
        train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUF, reshuffle_each_iteration=True)
        val_dataset = val_dataset.shuffle(buffer_size=SHUFFLE_BUF, reshuffle_each_iteration=True)

tokenizer = AlbertTokenizer.from_pretrained(PRETRAINED_PATH)
with open(PRETRAINED_PATH + 'config.json') as f:
    config = json.load(f)

# training configurations loop
for desired_batch_size, lr in itertools.product(DESIRED_BATCH_SIZES, LR):
    model = QaAlbertModel(config, PRETRAINED_PATH)
    linear_hidden_size = model.get_linear_hidden_size
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss=loss_fn, optimizer=optimizer)

    print(f"SETUP DONE - START TRAINING with batch size: {desired_batch_size} and lr: {lr}")
    weird_grads = False
    current_time = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d-%H%M%S")
    model_weights_path = MODEL_SAVE_PATH + 'lr{}_'.format(lr) + 'batch{}_'.format(desired_batch_size) + current_time
    log_dir = LOG_DIR + 'lr{}_'.format(lr) + 'batch{}_'.format(desired_batch_size) + current_time
    train_log_dir = log_dir + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = log_dir + '/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    mean_total_batch_time = 0
    """ 
    TODO:   - some answers reference a position in the question and don't change much from it --> train batch losses that don't go down
            - what about questions with no answers? 
            - for evaluation, compute score of all answer spans with s*t_i + e*t_j (s: start vector, e: end vector)
    """
    for e in range(EPOCHS):
        epoch_start = time.time()
        train_epoch_loss_avg = tf.keras.metrics.Mean()
        train_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        val_epoch_loss_avg = tf.keras.metrics.Mean()
        val_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_grads_avg = tf.keras.metrics.Mean()

        tape = tf.GradientTape()
        train_batch_times, val_batch_times, i, grads_idx = [], [], 0, 0
        accumulated_grads_0 = tf.zeros((config['hidden_size'], linear_hidden_size))
        accumulated_grads_1 = tf.zeros([linear_hidden_size])
        accumulated_grads_2 = tf.zeros((linear_hidden_size, config['num_labels']))
        accumulated_grads_3 = tf.zeros((config['num_labels']))
        train_len_data = int(USE * np.ceil(len(input_tokens_train) / desired_batch_size))
        take_train_data = int(USE * np.ceil(len(input_tokens_train) / BATCH_SIZE))
        val_len_data = int(USE * np.ceil(len(input_tokens_val) / desired_batch_size))
        take_val_data = int(USE * np.ceil(len(input_tokens_val) / BATCH_SIZE))
        total_len_data = train_len_data + val_len_data

        for inputs, labels in train_dataset.take(take_train_data):
            if grads_idx % desired_batch_size is 0:
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
                preds = model(inputs=inputs_pad, token_type_ids=token_type_ids, training=DROPOUT)[0]
                loss = loss_fn(y_pred=preds, y_true=labels)

                grads = tape.gradient(loss, model.trainable_variables)
                accumulated_grads_0 += grads[0]
                accumulated_grads_1 += grads[1]
                accumulated_grads_2 += grads[2]
                accumulated_grads_3 += grads[3]
                if grads_idx % desired_batch_size is 0 or int(grads_idx / BATCH_SIZE) is take_train_data:
                    optimizer.apply_gradients(zip([accumulated_grads_0, accumulated_grads_1, accumulated_grads_2, accumulated_grads_3], model.trainable_variables))

            # if e is 50 and (i is 1 or i is 2):
            #     print(f"Inputs 1):\n{tokenizer.decode(inputs_pad[0])}\nInputs 2):\n{tokenizer.decode(inputs_pad[1])}\nLabels:\n{labels}\nPreds: {tf.math.argmax(preds, axis=2)}")
            #     if grads_idx % desired_batch_size is 0:
            #         sys.exit()

            # Track progress
            train_epoch_loss_avg(loss)
            train_epoch_accuracy(labels, preds)
            epoch_grads_avg(tf.reduce_mean(tf.math.abs(accumulated_grads_0)) + tf.reduce_mean(tf.math.abs(accumulated_grads_2)))

            if grads_idx % desired_batch_size is 0 or int(grads_idx / BATCH_SIZE) is take_train_data:
                # check that avg grads don't vanish/explode
                if i % int(CHECK_AVG_LOSS * train_len_data) is 0:
                    grads_mean = epoch_grads_avg.result()
                    if grads_mean < GRADS_BOUNDS[0] or grads_mean > GRADS_BOUNDS[1]:
                        weird_grads = True
                        with train_summary_writer.as_default():
                            abort_statement = f'Due to grads being {grads_mean}, this configuration will be aborted at {current_time}.\n' + \
                                              f'LR: {lr} - EPOCH: {e} - ITER: {i} - BETAS: {BETA1}, {BETA2}\n'
                            print(abort_statement)
                            tf.summary.text('ABORT TRAINING', abort_statement, step=1.0)
                        break

                train_batch_times.append(time.time() - t)
                mean_total_batch_time = np.mean(train_batch_times) if mean_total_batch_time is 0 else mean_total_batch_time
                total_time_left = mean_total_batch_time * (total_len_data - i - 1 + (EPOCHS - e - 1) * total_len_data)
                print("Epoch {}: Train Step {} / {} took {:.3f}s (~{} left) - Train batch loss: {}".format(
                    e, i, total_len_data, train_batch_times[-1], get_hms_string(total_time_left), loss), end='\r')

                # set accumulated values to 0 and reset tape for next gradient update
                with tape:
                    tape.reset()
                accumulated_grads_0 = tf.zeros((config['hidden_size'], linear_hidden_size))
                accumulated_grads_1 = tf.zeros([linear_hidden_size])
                accumulated_grads_2 = tf.zeros((linear_hidden_size, config['num_labels']))
                accumulated_grads_3 = tf.zeros((config['num_labels']))

        i, grads_idx = 0, 0
        for inputs, labels in val_dataset.take(take_val_data):
            if grads_idx % desired_batch_size is 0:
                # time and index for desired batch
                i += 1
                t = time.time()
            grads_idx += BATCH_SIZE

            # due to difficulties with ragged tensors, pad inputs and transform them to regular tensor
            inputs_pad = inputs[0].to_tensor(default_value=0)
            token_type_ids = inputs[1].to_tensor(default_value=1)
            labels = labels.to_tensor()

            # Only pass data through model and compare prediction with labels
            preds = model(inputs=inputs_pad, token_type_ids=token_type_ids, training=False)[0]
            loss = loss_fn(y_pred=preds, y_true=labels)

            # Track progress
            val_epoch_loss_avg(loss)
            val_epoch_accuracy(labels, preds)
            val_batch_times.append(time.time() - t)
            if grads_idx % desired_batch_size is 0 or int(grads_idx / BATCH_SIZE) is take_val_data:
                mean_total_batch_time = np.mean(train_batch_times) + np.mean(val_batch_times)
                total_time_left = np.mean(val_batch_times) * val_len_data - i - 1 + mean_total_batch_time * (EPOCHS - e - 1) * total_len_data
                print("Epoch {}: Val Step {} / {} took {:.3f}s (~{} left) - Val batch loss: {}".format(
                    e, i, val_len_data, val_batch_times[-1], get_hms_string(total_time_left), loss), end='\r')

        # End epoch
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_epoch_loss_avg.result(), step=e)
            tf.summary.scalar('accuracy', train_epoch_accuracy.result(), step=e)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_epoch_loss_avg.result(), step=e)
            tf.summary.scalar('accuracy', val_epoch_accuracy.result(), step=e)
        print(f"Epoch {e} took {get_hms_string(time.time()-epoch_start)} with "
              f"train: loss: {train_epoch_loss_avg.result()} - acc: {train_epoch_accuracy.result()} :::: "
              f"val: loss: {val_epoch_loss_avg.result()} - acc: {val_epoch_accuracy.result()}")
        train_epoch_loss_avg.reset_states()
        train_epoch_accuracy.reset_states()
        val_epoch_loss_avg.reset_states()
        val_epoch_accuracy.reset_states()
        epoch_grads_avg.reset_states()
        if weird_grads is True:
            break

        if not os.path.exists(model_weights_path):
            os.makedirs(model_weights_path)
        model.save_weights(model_weights_path + '/model', save_format='tf')

