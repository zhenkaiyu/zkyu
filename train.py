#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime

import data_input_helper as data_helpers
from eval import eval
from text_cnn import TextCNN
import math
from tensorflow.contrib import learn
import csv
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("train_data_file", "/var/proj/sentiment_analysis/data/cutclean_tiny_stopword_corpus10000.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_data_file", "data1/senti_data.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
tf.flags.DEFINE_string("pre_emb_file", "data/fasttext_wiki.zh.txt", "w2v_file path")
# tf.flags.DEFINE_string("entity_file", "data/entity2.txt", "w2v_file path")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def load_data(w2v_model):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.train_data_file)
    # for x in x_text:
    #     l = len(x.split(" "))
    #     break

    max_document_length = max([len(x.split(" ")) for x in x_text])
    print ('len(x) = ',len(x_text),' ',len(y))
    print(' max_document_length = ' , max_document_length)

    x = []
    vocab_size = 0
    if(w2v_model is None):
      vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
      x = np.array(list(vocab_processor.fit_transform(x_text)))
      vocab_size = len(vocab_processor.vocabulary_)

      # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time()))))
      vocab_processor.save("vocab.txt")
      print( 'save vocab.txt')
    else:
      x = data_helpers.get_text_idx(x_text,y,w2v_model.vocab,max_document_length)
      vocab_size = len(w2v_model.vocab)
      print('use w2v .bin')

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # entity_shuffled = entity[shuffle_indices]
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    # entity_train ,entity_dev = entity_shuffled[:dev_sample_index], entity_shuffled[dev_sample_index:]
    return x_train,x_dev,y_train,y_dev,vocab_size




def train(w2v_model):
    # Training
    # ==================================================
    x_train, x_dev, y_train, y_dev ,vocab_size = load_data(w2v_model)
    best_accurcy = 0
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                w2v_model,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -5, 5), v]
                                 for g, v in grads_and_vars]
            train_op = optimizer.apply_gradients(capped_grads_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            #train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """

                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  # cnn.entity:  entity_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # _, step, summaries, loss, accuracy,(w,idx) = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,cnn.get_w2v_W()],
                #     feed_dict)
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print w[:2],idx[:2]
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch,writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  # cnn.entity : entity_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy


            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)


            def dev_test():
                # batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev,entity_dev)), FLAGS.batch_size, 1)
                # for batch_dev in batches_dev:
                #     x_batch_dev, y_batch_dev ,entity_batch_dev= zip(*batch_dev)
                accuracy = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                return accuracy
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                # Training loop. For each batch...
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    accuracy = dev_test()
                    if accuracy > best_accurcy:
                        best_accurcy = accuracy
                        print("bset_accuracy {}\n".format(best_accurcy))
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        path_pass = str(path)
                        path_pass = path_pass.split('\\')
                        print(path_pass)
                        print("Saved model checkpoint to {}\n".format(path))
    return x_dev,y_dev,path_pass

if __name__ == "__main__":
    w2v_wr = data_helpers.w2v_wrapper(FLAGS.pre_emb_file)
    x_dev,y_dev,path = train(w2v_wr.model)
    eval(x_dev,y_dev,path)