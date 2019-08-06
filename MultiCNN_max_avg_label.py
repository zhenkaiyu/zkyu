#!/usr/bin/env python
# encoding: utf-8

"""
@author: izyq
@file: MultiCNN.py
@time: 2018/11/20 14:18
"""
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers import fully_connected
import numpy as np
import math
from keras.utils.np_utils import to_categorical
import os
from sklearn.metrics import accuracy_score
from BatchUtil import batch_generator_test,batch_generator
from gensim.models import KeyedVectors

os.environ['CUDA_VISIBLE_DEVICES']='2'
class Multicnns:
    def __init__(self,config,naive):
        self.sequence_length = config['sequence_length']
        self.num_classes = config['num_classes']

        self.embedding_weight = config['embedding_weight']
        self.filters_sizes=config['filters_size']
        self.num_filters = config['num_filters']

        self.dropout_rate_embed = config['dropout_rate_embed']
        self.dropout_rate_fully = config['dropout_rate_fully']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']
        self.train_max_patience = config['train_max_patience']

        # self.margin=config['margin']

        # self.label_dim=config['label_dim']

        self.naive=naive

        self.build()
        pass

    def conv2d(self,scope_name,input_x, kernel,in_channels,out_channels,strides,padding,residual):
        with tf.name_scope(scope_name):
            # if self.naive:
            #     output_channel=out_channels
            # else:
            #     output_channel=out_channels*2
            # 参数太多容易过拟合
            output_channel=out_channels
            filter_shape=[kernel[0],kernel[1],in_channels,output_channel]
            # initializers.xavier_initializer()
            W=tf.Variable(initial_value=initializers.xavier_initializer()(filter_shape),dtype=tf.float32,name='W')
            b = tf.Variable(initial_value=initializers.xavier_initializer()([output_channel,]), dtype=tf.float32, name='b')
            conv=tf.nn.conv2d(input_x,W,strides=[1,strides[0],strides[1],1],padding=padding)
            conv_b=tf.nn.bias_add(conv,b)
            # conv_g=conv_b[:,:,:,:out_channels]*tf.sigmoid(conv_b[:,:,:,out_channels:])
            # (bs,sl,1,d)
            if residual:
                # (bs,sl,1,d) (bs,sl,d,1)
                return conv_b+tf.transpose(input_x,[0,1,3,2])
            else:
                return conv_b


    def get_attention(self,inputx,output_size):
        attention_context_vector=tf.Variable(name='atten_context_vetor',initial_value=initializers.xavier_initializer()([output_size,]),dtype=tf.float32)
        input_projection=fully_connected(inputx,output_size)
        # (bs,sl,o) (o,)=>(bs,sl)
        vector_attn=tf.reduce_sum(tf.multiply(input_projection,attention_context_vector),axis=-1)
        # (bs,sl)
        attention_weight=tf.nn.softmax(vector_attn)
        attention_weight_expand=tf.expand_dims(attention_weight,-1)
        weighted_projection=tf.multiply(input_projection,attention_weight_expand)
        outputs=tf.reduce_sum(weighted_projection,1)
        # (bs,output_size)
        return outputs

    def get_gate(self,x,y):
        # (bs,4d)
        xy=tf.concat([x,y],-1,name='xy')
        # (bs,1)
        gate=tf.sigmoid(fully_connected(xy,1),name='gate')

        # # (4d,1)
        # w=tf.Variable(initial_value=initializers.xavier_initializer()([4*self.num_hidden,1]),dtype=tf.float32,name='w')
        # 做一个sigmoid
        # gate=tf.sigmoid(tf.matmul(xy,w),name='gate')
        return gate

    def build(self):
        with tf.name_scope('placeholder'):
            self.input_ph=tf.placeholder(dtype=tf.int32,shape=(None,self.sequence_length),name='input')
            self.label_ph = tf.placeholder(tf.float32, shape=(None, self.num_classes), name='label')

            self.lr_ph=tf.placeholder(dtype=tf.float32,name='learning_rate')
            self.dropout_rate_embed_ph = tf.placeholder(dtype=tf.float32, name='dropout_rate_embed')
            self.dropout_rate_fully_ph = tf.placeholder(dtype=tf.float32, name='dropout_rate_fully')

        with tf.name_scope('embedding'):
            self.embedding_weight_v=tf.get_variable("word_embeddings",
                    initializer=self.embedding_weight)
            self.feature_embedding = tf.nn.embedding_lookup(self.embedding_weight_v, self.input_ph)
            # (bs,sl,d,1)
            feature_embedding_expand = tf.expand_dims(self.feature_embedding, axis=-1)
            # 此处的dropout表示保留的比例
            self.feature_embedding_expand=tf.nn.dropout(feature_embedding_expand,self.dropout_rate_embed_ph)
            # self.label_embedding=tf.Variable(initial_value=initializers.xavier_initializer()([self.num_classes,self.label_dim]),dtype=tf.float32,name='label_embedding')

        self.embedding_size=self.embedding_weight.shape[1]
        feature_maps=[]
        for fs in self.filters_sizes:
            with tf.name_scope('fs_conv_%s'%fs):
                # (bs,sl,e_size) if naive (bs,)
                # padding='same'时,注意滑动窗口的设置,width=embedding_size
                conv=self.conv2d('fs_%s'%fs,self.feature_embedding_expand,[fs,self.embedding_size],1,self.embedding_size,[1,self.embedding_size],'SAME',True)
                # (bs,sl,1,out)
                conv=tf.nn.relu(conv)
                # (bs,1,1,out)
                max_pool=tf.nn.max_pool(conv,[1,self.sequence_length,1,1],[1,1,1,1],'VALID')
                avg_pool=tf.nn.avg_pool(conv,[1,self.sequence_length,1,1],[1,1,1,1],'VALID')
                # (bs,out)
                # weighted_output=self.get_attention(tf.squeeze(conv,-2),self.embedding_size)
                # # (bs,1,1,out)
                # weighted_output=tf.reshape(weighted_output,(-1,1,1,self.embedding_size))
                # (bs,1,1,out*3)
                feature_maps.append(tf.concat([max_pool,avg_pool],-1,name='max_avg'))

                # (bs,out)
                # feature_maps.append(weighted_output)
        # (bs,3*out*2)
        self.feature_map=tf.concat(feature_maps,-1)
        nb_featuresize=6
        if self.naive:
            self.feature_map=tf.reshape(self.feature_map,[-1,nb_featuresize*self.embedding_size])
        print(self.feature_map.shape)

        # with tf.name_scope('label_sim'):
        #     # (la,d')=>(la,3d) 对标签做一个维度映射
        #     self.label_projection=fully_connected(self.label_embedding,self.embedding_size*3)
        #     # (bs,3d) (la,3d)+>(bs,la) 每一句话与所有标签做一个相似度计算
        #     sim=tf.matmul(self.feature_map,self.label_projection,transpose_b=True)
        #     # (bs,la) (la,3d) 标签向量
        #     self.label_weight=tf.matmul(sim,self.label_projection)


        with tf.name_scope('dropout'):
            # (bs,3d)
            # merge_o=tf.concat([self.feature_map,self.label_weight],-1)
            # gate=self.get_gate(self.feature_map,self.label_weight)
            # merge_o=tf.add(gate*self.feature_map,(1-gate)*self.label_weight,name='merge_feature_label')
            # print('----merge_o------')
            # print(merge_o)
            # fm_dp1=tf.nn.dropout(self.feature_map,self.dropout_rate_fully_ph)
            # dense1=fully_connected(fm_dp1,self.embedding_size*2)
            fm_dp2=tf.nn.dropout(self.feature_map,self.dropout_rate_fully_ph)

        with tf.name_scope('outputs'):
            self.sm_W = tf.Variable(initial_value=initializers.xavier_initializer()([self.embedding_size*nb_featuresize,self.num_classes]), dtype=tf.float32, name='sm_W')
            self.sm_b = tf.Variable(initial_value=initializers.xavier_initializer()([self.num_classes,]), dtype=tf.float32, name='sm_b')
            # (bs,num_classes)
            self.logits=tf.nn.xw_plus_b(fm_dp2,self.sm_W,self.sm_b,name='logits')
            # self.logits_sm=tf.nn.softmax(self.logits,-1)
            # (bs,)
            preds=tf.argmax(self.logits,-1)

        with tf.name_scope('loss'):
            # 修正交叉熵损失函数 (bs,nc),(0.6)=>(bs,nc)
            # self.residual=self.logits_sm-self.margin
            # 只有在正确位置上的值才不为零，相当于one hot矩阵,=>求和，降维(bs,)
            # self.right_pos=tf.reduce_sum(self.residual*self.label_ph,-1)
            # self.new_lambda=tf.cast(tf.less(self.right_pos,0),dtype=tf.float32,name='new_lambda')
            # self.new_loss=tf.reduce_mean(self.new_lambda*tf.nn.softmax_cross_entropy_with_logits(labels=self.label_ph,
            #                                                                                      logits=self.logits),name='new_loss')
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_ph,logits=self.logits),name='loss')

        with tf.name_scope('acc'):
            equal=tf.equal(preds,tf.argmax(self.label_ph,-1),name='equal')
            self.acc=tf.reduce_mean(tf.cast(equal,tf.float32),name='acc')

        with tf.name_scope('train'):
            self.global_step=tf.Variable(0,trainable=False,dtype=tf.int32,name='global_step')
            optimizer=tf.train.AdamOptimizer(self.lr_ph)
            # grads_and_vars=optimizer.compute_gradients(self.new_loss)
            grads_and_vars=optimizer.compute_gradients(self.loss)
            self.train_op=optimizer.apply_gradients(grads_and_vars,global_step=self.global_step)

        self.init_var=tf.global_variables_initializer()
        self.saver=tf.train.Saver()

    def fit(self,x_train,y_train,x_dev,y_dev):
        max_acc = 0.
        current_patience = 0

        data_count = len(x_train)
        print('the size of x_train data:', data_count)
        print('the size of test data:',len(x_dev))

        batches = batch_generator(x_train, y_train, self.batch_size)
        batch_num_in_epoch = int(math.ceil(data_count / float(self.batch_size)))

        tf_config=tf.ConfigProto()
        tf_config.gpu_options.allow_growth=True

        with tf.Session() as sess:
            # 初始化variable(不使用训练好的参数)
            sess.run(self.init_var)
            for step in range(self.num_epochs):
                # if step<1:
                #     lr_ph=self.learning_rate
                # elif step<3:
                #     lr_ph=1e-4
                # else:
                #     lr_ph=1e-5
                lr_ph = self.learning_rate
                print('Epoch %d / %d:' % (step + 1, self.num_epochs),'learning_rate= %e '%(lr_ph))
                train_loss = 0.0
                train_acc = 0.0

                for _ in range(batch_num_in_epoch):
                    x_batch, y_batch = batches.__next__()
                    feed_dict = dict()
                    feed_dict[self.input_ph] = x_batch
                    # feed_dict[self.sequence_actual_lengths_ph]=np.array([list(x).index(0)+1 for x in x_batch])
                    feed_dict[self.label_ph] = y_batch
                    feed_dict[self.dropout_rate_embed_ph] = self.dropout_rate_embed
                    feed_dict[self.dropout_rate_fully_ph]=self.dropout_rate_fully
                    feed_dict[self.lr_ph]=lr_ph

                    _, loss_d_a, acc_, global_step_ = sess.run(
                        [self.train_op, self.loss, self.acc, self.global_step], feed_dict=feed_dict)
                    train_loss += loss_d_a
                    train_acc += acc_

                    if global_step_%100==0:

                        # acc_dev=self.predict(x_dev,y_dev,self.batch_size,sess,self.score)
                        acc_dev=self.predict(x_dev,y_dev,self.batch_size,sess,self.logits)

                        if max_acc<acc_dev:
                            max_acc=acc_dev
                            current_patience=0
                            self.saver.save(sess,self.model_path)
                            print('%d in training %f'%(step+1,max_acc))
                            print('model has saved to %s'%self.model_path)
                            pass

                print('%d epoch over--------------------------------------------'%(step+1))
                train_loss/=float(batch_num_in_epoch)
                train_acc/=float(batch_num_in_epoch)
                print('train_loss: ',train_loss)
                print('train_acc: ',train_acc)

                acc_train=self.predict(x_train,y_train,self.batch_size,sess,self.logits)
                print('train set acc=',acc_train)

                acc_dev=self.predict(x_dev,y_dev,self.batch_size,sess,self.logits)
                print('dev set acc=',acc_dev)

                if max_acc<acc_dev:
                    max_acc=acc_dev
                    self.saver.save(sess,self.model_path)
                    print('%d model has savd'%(step+1))
                else:
                    current_patience+=1
                    if current_patience>=self.train_max_patience:
                        print('提前终止了！！！！！')
                        return

    def predict(self,x_data,y,batch_size,sess,scores):
        all_predict=[]
        data_count=len(x_data)

        batches=batch_generator_test(x_data,y,self.batch_size)
        batch_num_in_epochs=int(math.ceil(data_count/float(self.batch_size)))

        for _ in range(batch_num_in_epochs):
            x_batch,y_batch=batches.__next__()
            feed_dict=dict()
            feed_dict[self.input_ph]=x_batch
            # feed_dict[self.sequence_actual_lengths_ph]=np.array([list(x).index(0)+1 for x in x_batch])
            feed_dict[self.label_ph]=y_batch
            feed_dict[self.dropout_rate_embed_ph]=1.0
            feed_dict[self.dropout_rate_fully_ph]=1.0

            logits=sess.run(scores,feed_dict=feed_dict)

            # test_loss+=loss_d_a
            predicts=np.argmax(logits,axis=-1)
            all_predict.extend(predicts)

        y=np.argmax(y,axis=-1)
        recall_s=accuracy_score(y_true=y,y_pred=all_predict)

        return recall_s

if __name__=='__main__':
    # chinese
    language='chinese'
    navie=True
    # file='multicnn_new_npy'

    # ch_rpath='/home/fhzhu/youqianzhu/newstitle/data/nlpcc_data/processed_word/ltp_new/npy/'
    ch_rpath='/home/fhzhu/youqianzhu/newstitle/data/nlpcc_data/processed_word/new_npy/'
    en_rpath='data/'
    tag_path='data/'
    config = dict()
    # labelbased/cnn_model/bestmodel
    # config['num_classes']=18
    # config['label_dim']=128
    #
    # config['filters_size']=[3,4,5]
    #
    # config['dropout_rate_embed']=0.8
    # config['dropout_rate_fully']=0.5
    # config['learning_rate']=1e-3
    # config['batch_size']=64
    # config['num_epochs']=16
    # if step<2:
    #     lr_ph=self.learning_rate
    # else:
    #     lr_ph=1e-4

    config['num_classes'] = 2
    # config['label_dim']=256

    config['filters_size'] = [3,4,5]

    config['dropout_rate_embed'] = 0.8
    config['dropout_rate_fully'] = 0.5
    config['learning_rate'] = 1e-3
    config['batch_size'] = 64
    config['num_epochs'] = 100
    # config['margin']=0.6

    if language=='chinese':
        EMBEDDING_SIZE = 300
        # embedding=np.array(np.load(os.path.join(en_rpath,'glove.6B.100d.txt')),dtype=np.float32)
        embedding_file = "data/fasttext_wiki.zh.txt"
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=False, unicode_errors='ignore')
        i = 0
        for key in model.vocab:
            model.vocab[key] = i
            i = i + 1
        if 'unknown' not in model.vocab:
            unknown_vec = np.random.uniform(-0.1, 0.1, size=100)
            model.vocab['unknown'] = len(model.vocab)
            model.vectors = np.row_stack((model.vectors, unknown_vec))

    else:

        EMBEDDING_SIZE=300

        # embedding=np.array(np.load(os.path.join(en_rpath,'glove.6B.100d.txt')),dtype=np.float32)
        embedding_file = "data1/glove.6B.100d.txt"
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=False, unicode_errors='ignore')
        i = 0
        for key in model.vocab:
            model.vocab[key] = i
            i = i + 1
        if 'unknown' not in model.vocab:
            unknown_vec = np.random.uniform(-0.1, 0.1, size=100)
            model.vocab['unknown'] = len(model.vocab)
            model.vectors = np.row_stack((model.vectors, unknown_vec))



        config['sequence_length']=474
        config['model_path']='data/model'
        x_train = np.load(os.path.join(en_rpath, 'x_train.npy'))
        x_dev=np.load(os.path.join(en_rpath,'x_dev.npy'))

    y_train = np.load(os.path.join(tag_path, 'y_train.npy'))
    # y_train = to_categorical(y_train, num_classes=config['num_classes'])
    y_dev=np.load(os.path.join(tag_path,'y_dev.npy'))
    print(x_train)
    print(y_train)
    # y_dev=to_categorical(y_dev,num_classes=config['num_classes'])
    config['embedding_weight']=model.vectors.astype(np.float32)
    config['num_filters']=EMBEDDING_SIZE
    # config['model_path']='/home/fhzhu/youqianzhu/newstitle/model/english_multicnn_naive/model/best_model'

    # /home/fhzhu/youqianzhu/newstitle/model/english_multicnn_glu_attn/model/
    config['train_max_patience']=100

    # 是否conv+maxpooling
    multicnn=Multicnns(config,navie)
    multicnn.fit(x_train,y_train,x_dev,y_dev)