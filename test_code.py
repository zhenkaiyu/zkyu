import tensorflow as tf
from model import Multicnns
from BatchUtil import batch_generator_test,batch_generator
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
import math
import numpy as np

def test(x_test,y_test,entity_test,config,naive):
    model = Multicnns(config,naive)
    data_count = len(x_test)
    batches = batch_generator_test(x_test, y_test,entity_test,64)
    batch_num_in_epochs = int(math.ceil(data_count / float(64)))
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    all_predict = []

    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.latest_checkpoint('data/model/duanluo_tezheng')
        print("+++++++++++++++++++")
        saver.restore(sess,ckpt)
        print("loading model!")
        tf.summary.FileWriter('data/log')
        test_accuracy_list = []
        test_loss_list = []
        for j in range(batch_num_in_epochs):
            x_batch, y_batch, entity_batch = batches.__next__()
            feed_dict = dict()
            feed_dict[model.input_ph] = x_batch
            # feed_dict[self.sequence_actual_lengths_ph]=np.array([list(x).index(0)+1 for x in x_batch])
            feed_dict[model.label_ph] = y_batch
            feed_dict[model.entity_ph] = entity_batch
            feed_dict[model.dropout_rate_embed_ph] = 1.0
            feed_dict[model.dropout_rate_fully_ph] = 1.0

            logits = sess.run(model.logits, feed_dict=feed_dict)
            # test_loss+=loss_d_a
            predicts = np.argmax(logits,axis=-1)
            all_predict.extend(predicts)
        y = np.argmax(y_test,axis=-1)
        recall_s = accuracy_score(y_true=y, y_pred=all_predict)
        return recall_s, all_predict, y





if __name__ == '__main__':
    navie = True
    config = dict()
    embedding_file = "data/glove.6B.100d.txt"
    model_emd = KeyedVectors.load_word2vec_format(embedding_file, binary=False, unicode_errors='ignore')

    config['num_classes'] = 2
    # config['label_dim']=256

    config['filters_size'] = [3, 4, 5]

    config['dropout_rate_embed'] = 0.8
    config['dropout_rate_fully'] = 0.5
    config['learning_rate'] = 1e-3
    config['batch_size'] = 64
    config['num_epochs'] = 50
    config['sequence_length'] = 474
    config['model_path'] = 'data/model/'
    config['embedding_weight'] = model_emd.vectors.astype(np.float32)
    config['num_filters'] = 300
    # config['model_path']='/home/fhzhu/youqianzhu/newstitle/model/english_multicnn_naive/model/best_model'

    # /home/fhzhu/youqianzhu/newstitle/model/english_multicnn_glu_attn/model/
    config['train_max_patience'] = 100

    x_test = np.load('data/x_dev.npy')
    y_test = np.load('data/y_dev.npy')
    entity_test = np.load('data/entity_dev.npy')
    acc_dev,y_pre,y= test(x_test,y_test,entity_test,config,navie)

    print("test_acc:")
    print(acc_dev)

