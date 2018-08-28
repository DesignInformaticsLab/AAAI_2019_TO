import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import random
import matlab.engine

class TO_generator():
    def __init__(self):
        # Input parameter
        self.nelx, self.nely = 12 * 10, 4 * 10
        self.nn = self.nelx * self.nely
        self.batch_size = 10
        self.initial_num = 100
        self.Prepared_training_sample = True  # True if samples are pre-solved offline

        # network parameter
        self.z_dim = 2 * 41 * 41
        self.width = self.nely
        self.height = self.nelx
        self.h_dim = self.width / 8 * self.height / 8

        self.deconv2_1_features = 32 * 3
        self.deconv2_2_features = 32 * 3
        self.deconv3_1_features = 32 * 2
        self.deconv3_2_features = 32 * 2
        self.deconv4_1_features = 32
        self.deconv4_2_features = 32

        # log dir
        self.directory_data='experiment_data'
        self.directory_model='model_save'
        self.directory_result='experiment_result'
        self.directory_model_3D = 'model_save_3D'
        self.directory_model_3D_final = 'Final_3D'


    def xavier_init(self, dim_size):
        in_dim = dim_size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=dim_size, stddev=xavier_stddev)


    def P(self, z, pretrained_weights, vanilla_weights):

        weights = vanilla_weights
        weights = pretrained_weights

        h1 = tf.nn.relu(tf.matmul(z, self.P_W1) + self.P_b1)
        h1 = tf.reshape(h1, [self.batch_size, self.width / 8, self.height / 8, 1])
        h2_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h1, weights['deconv2_1_weight'], strides=[1, 1, 1, 1], padding='SAME',
                                           output_shape=[self.batch_size, self.width/8, self.height/8, self.deconv2_1_features]), weights['deconv2_1_bias']))

        h2_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h2_1, weights['deconv2_2_weight'], strides=[1, 2, 2, 1], padding='SAME',
                                           output_shape=[self.batch_size, self.width/4, self.height/4, self.deconv2_2_features]), weights['deconv2_2_bias']))

        h3_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h2_2, weights['deconv3_1_weight'], strides=[1, 1, 1, 1], padding='SAME',
                                           output_shape=[self.batch_size, self.width/4, self.height/4, self.deconv3_1_features]), weights['deconv3_1_bias']))

        h3_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h3_1, weights['deconv3_2_weight'], strides=[1, 2, 2, 1], padding='SAME',
                                           output_shape=[self.batch_size, self.width/2, self.height/2, self.deconv3_2_features]), weights['deconv3_2_bias']))

        h4_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h3_2, weights['deconv4_1_weight'], strides=[1, 1, 1, 1], padding='SAME',
                                           output_shape=[self.batch_size, self.width/2, self.height/2, self.deconv4_1_features]), weights['deconv4_1_bias']))

        h4_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h4_1, weights['deconv4_2_weight'], strides=[1, 2, 2, 1], padding='SAME',
                                           output_shape=[self.batch_size, self.width/1, self.height/1, self.deconv4_2_features]), weights['deconv4_2_bias']))

        h5 = (tf.add(tf.nn.conv2d_transpose(h4_2, weights['deconv5_weight'], strides=[1, 1, 1, 1], padding='SAME',
                                            output_shape=[self.batch_size, self.width/1, self.height/1, 1]), weights['deconv5_bias']))

        prob = tf.nn.sigmoid(h5)
        return prob

    def init_train(self):

        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        saver = tf.train.import_meta_graph('./Final_1D/model_1D_final.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./Final_1D/'))
        self.graph = tf.get_default_graph()
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):
        self.F_input = tf.placeholder(tf.float32, shape=([self.batch_size, self.z_dim]))

        pretrained_weights = {}
        P_W1_direction = self.sess.run('P_W1:0')
        self.P_W1 = tf.Variable(P_W1_direction, dtype=tf.float32)
        P_b1 =  self.sess.run('P_b1:0')
        self.P_b1 = tf.Variable(P_b1, dtype = tf.float32)
        pretrained_weights['deconv2_1_weight'] = tf.Variable(self.sess.run('deconv2_1_weight:0'),dtype = tf.float32)
        pretrained_weights['deconv2_1_bias'] = tf.Variable(self.sess.run('deconv2_1_bias:0'),dtype = tf.float32)
        pretrained_weights['deconv2_2_weight'] = tf.Variable(self.sess.run('deconv2_2_weight:0'),dtype = tf.float32)
        pretrained_weights['deconv2_2_bias'] = tf.Variable(self.sess.run('deconv2_2_bias:0'),dtype = tf.float32)
        pretrained_weights['deconv3_1_weight'] = tf.Variable(self.sess.run('deconv3_1_weight:0'),dtype = tf.float32)
        pretrained_weights['deconv3_1_bias'] = tf.Variable(self.sess.run('deconv3_1_bias:0'),dtype = tf.float32)
        pretrained_weights['deconv3_2_weight'] = tf.Variable(self.sess.run('deconv3_2_weight:0'),dtype = tf.float32)
        pretrained_weights['deconv3_2_bias'] = tf.Variable(self.sess.run('deconv3_2_bias:0'),dtype = tf.float32)
        pretrained_weights['deconv4_1_weight'] = tf.Variable(self.sess.run('deconv4_1_weight:0'),dtype = tf.float32)
        pretrained_weights['deconv4_1_bias'] = tf.Variable(self.sess.run('deconv4_1_bias:0'),dtype = tf.float32)
        pretrained_weights['deconv4_2_weight'] = tf.Variable(self.sess.run('deconv4_2_weight:0'),dtype = tf.float32)
        pretrained_weights['deconv4_2_bias'] = tf.Variable(self.sess.run('deconv4_2_bias:0'),dtype = tf.float32)
        pretrained_weights['deconv5_weight'] = tf.Variable(self.sess.run('deconv5_weight:0'),dtype = tf.float32)
        pretrained_weights['deconv5_bias'] = tf.Variable(self.sess.run('deconv5_bias:0'),dtype = tf.float32)
        self.pretrained_weights = pretrained_weights

        vanilla_weights = {}
        vanilla_weights['deconv2_1_weight'] = tf.Variable(tf.truncated_normal([4, 4, self.deconv2_1_features, 1], stddev=0.1, dtype=tf.float32))
        vanilla_weights['deconv2_1_bias'] = tf.Variable(tf.zeros([self.deconv2_1_features], dtype=tf.float32))
        vanilla_weights['deconv2_2_weight'] = tf.Variable(tf.truncated_normal([4, 4, self.deconv2_2_features, self.deconv2_1_features], stddev=0.1, dtype=tf.float32))
        vanilla_weights['deconv2_2_bias'] = tf.Variable(tf.zeros([self.deconv2_2_features], dtype=tf.float32))
        vanilla_weights['deconv3_1_weight'] = tf.Variable(tf.truncated_normal([4, 4, self.deconv3_1_features, self.deconv2_2_features], stddev=0.1, dtype=tf.float32))
        vanilla_weights['deconv3_1_bias'] = tf.Variable(tf.zeros([self.deconv3_1_features], dtype=tf.float32))
        vanilla_weights['deconv3_2_weight'] = tf.Variable(tf.truncated_normal([4, 4, self.deconv3_2_features, self.deconv3_1_features], stddev=0.1, dtype=tf.float32))
        vanilla_weights['deconv3_2_bias'] = tf.Variable(tf.zeros([self.deconv3_2_features], dtype=tf.float32))
        vanilla_weights['deconv4_1_weight'] = tf.Variable(tf.truncated_normal([4, 4, self.deconv4_1_features, self.deconv3_2_features], stddev=0.1, dtype=tf.float32))
        vanilla_weights['deconv4_1_bias'] = tf.Variable(tf.zeros([self.deconv4_1_features], dtype=tf.float32))
        vanilla_weights['deconv4_2_weight'] = tf.Variable(tf.truncated_normal([8, 8, self.deconv4_2_features, self.deconv4_1_features], stddev=0.1, dtype=tf.float32))
        vanilla_weights['deconv4_2_bias'] = tf.Variable(tf.zeros([self.deconv4_2_features], dtype=tf.float32))
        vanilla_weights['deconv5_weight'] = tf.Variable(tf.truncated_normal([8, 8, 1, self.deconv4_2_features], stddev=0.1, dtype=tf.float32))
        vanilla_weights['deconv5_bias'] = tf.Variable(tf.zeros([1], dtype=tf.float32))
        self.vanilla_weights = vanilla_weights

        P_output = self.P(self.F_input, self.pretrained_weights, self.vanilla_weights)
        self.phi_true = tf.transpose(tf.reshape(P_output,[self.batch_size, self.nn]))

        self.starter_learning_rate=0.005
        self.global_step=tf.Variable(0,trainable=False)
        self.learning_rate=tf.train.exponential_decay(self.starter_learning_rate, self.global_step,1000,0.98,staircase=True)
        self.y_output=tf.placeholder(tf.float32, shape=([self.nn, self.batch_size]))
        self.recon_loss = tf.reduce_sum((self.phi_true-self.y_output)**2)/self.batch_size
        self.solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.recon_loss, self.global_step)
        self.sess.run(tf.global_variables_initializer())

    def load_initial_points(self):
        # pre-sampling the loading condition offline
        LHS = sio.loadmat('{}/LHS_train.mat'.format(self.directory_data))['LHS_train']

        LHS[:, 0] = LHS[:, 0] - 81
        LHS[:, 1] = LHS[:, 1] - 1
        LHS_x = np.int32(LHS[:, 0])
        LHS_y = np.int32(LHS[:, 1])
        LHS_z = LHS[:, 2]

        # force = -1
        # F_batch = np.zeros([len(LHS), self.z_dim])
        # for i in range(len(LHS)):
        #     Fx = force * np.sin(LHS_z[i])
        #     Fy = force * np.cos(LHS_z[i])
        #     F_batch[i, 2 * ((self.nely + 1) * LHS_x[i] + LHS_y[i] + 1) - 1] = Fy
        #     F_batch[i, 2 * ((self.nely + 1) * LHS_x[i] + LHS_y[i] + 1) - 2] = Fx
        return LHS, LHS_x, LHS_y, LHS_z

    def load_test_data(self):
        Y_test = sio.loadmat('{}/phi_true_test2.mat'.format(self.directory_data))['phi_true_test']  # prepared off-line
        test_load = sio.loadmat('{}/LHS_test2.mat'.format(self.directory_data))['LHS_test']  # prepared off-line
        test_load[:, 0] = test_load[:, 0] - 81
        test_load[:, 1] = test_load[:, 1] - 1

        test_load_x = np.int32(test_load[:, 0])
        test_load_y = np.int32(test_load[:, 1])
        test_load_z = test_load[:, 2]
        return Y_test, test_load_x, test_load_y, test_load_z

    def train_model(self):

        # generating initial points
        LHS, LHS_x, LHS_y, LHS_z = self.load_initial_points()
        Y_test, test_load_x, test_load_y, test_load_z = self.load_test_data()

        index_ind = random.sample(range(0, len(LHS)), self.initial_num)  # initial start with 100, can be modified

        if self.Prepared_training_sample == True:
            pass
        else:
            if not os.path.exists(self.directory_result):
                os.makedirs(self.directory_result)
            sio.savemat('{}/index_ind.mat'.format(self.directory_result), {'index_ind': index_ind})
            eng = matlab.engine.start_matlab()
            eng.infill_high_dim(1, nargout=0)

        budget = 0
        error_progress = []
        final_error = float('inf')
        terminate_step = 451
        starting_loss = 100
        decay_rate = 0.6

        def get_new_point(index_ind):
            try:
                add_point_index = sio.loadmat('{}/add_point_index.mat'.format(self.directory_result))['add_point_index'][0]
                index_ind = list(add_point_index) + index_ind
            except:
                print("add_point_index bug")
                pass
            return index_ind
        def get_Y_F():
            Y_train = sio.loadmat('{}/phi_true_train.mat'.format(self.directory_data))['phi_true_train']
            F_batch = np.zeros([len(LHS), self.z_dim])
            for i in range(len(LHS)):
                F_batch[i, 0] = LHS_x[i]
                F_batch[i, 1] = LHS_y[i]
                F_batch[i, 2] = LHS_z[i]
            force = -1
            F_batch = np.zeros([len(LHS), self.z_dim])
            for i in range(len(LHS)):
                Fx = force * np.sin(LHS_z[i])
                Fy = force * np.cos(LHS_z[i])
                F_batch[i, 2 * ((self.nely + 1) * LHS_x[i] + LHS_y[i] + 1) - 1] = Fy
                F_batch[i, 2 * ((self.nely + 1) * LHS_x[i] + LHS_y[i] + 1) - 2] = Fx
            return Y_train, F_batch

        # one-shot algorithm
        while len(index_ind) <= terminate_step:
            print("requirement doesn't match, current final_error={}, keep sampling".format(final_error))
            index_ind = get_new_point(index_ind)

            loss_list = []
            global_step_loss = len(index_ind) - self.initial_num
            decay_loss = starting_loss * decay_rate ** global_step_loss

            Y_train, F_batch = get_Y_F()

            for it in range(100000):
                random_ind = np.random.choice(index_ind, self.batch_size, replace=False)

                _, error = self.sess.run([self.solver, self.recon_loss],
                                    feed_dict={self.y_output: Y_train[random_ind].T, self.F_input: F_batch[random_ind]})
                # track training process
                if it % 100 == 0:
                    print('iteration:{}, recon_loss:{}, number of data used is:{}'.format(it, error, (len(index_ind))))
                if len(index_ind) % 100 == 0:  # plot loss curve
                    loss_list.append(error)
                # save model and exist this round of iteration
                if error <= decay_loss or error <= 1:
                    print('loss threshold is: {}'.format(decay_loss))
                    if not os.path.exists(self.directory_model):
                        os.makedirs(self.directory_model)
                    saver = tf.train.Saver()
                    saver.save(self.sess, '{}/model_sample_{}'.format(self.directory_model_3D, len(index_ind)))
                    print('converges, saving the model.....')
                    break
            print('number of data used is:{}'.format(len(index_ind)))

            def get_candidate():
                candidate_pool = list(
                    set(list(np.int32(np.linspace(0, len(Y_train) - 1, len(Y_train))))) - set(index_ind))
                random_candidate = np.random.choice(candidate_pool, 100, replace=False)
                LHS_candidate = sio.loadmat('{}/LHS_train.mat'.format(self.directory_data))['LHS_train'][
                    random_candidate]
                LHS_candidate[:, 0] = LHS_candidate[:, 0] - 81
                LHS_candidate[:, 1] = LHS_candidate[:, 1] - 1

                force = -1
                F_batch_candidate = np.zeros([len(LHS_candidate), self.z_dim])
                for i in range(len(LHS_candidate)):
                    Fx = force * np.sin(LHS_z[i])
                    Fy = force * np.cos(LHS_z[i])
                    F_batch_candidate[i, 2 * ((self.nely + 1) * LHS_x[i] + LHS_y[i] + 1) - 1] = Fy
                    F_batch_candidate[i, 2 * ((self.nely + 1) * LHS_x[i] + LHS_y[i] + 1) - 2] = Fx
                return  LHS_candidate, F_batch_candidate, random_candidate

            def get_test_load():
                LHS_candidate, F_batch_candidate, random_candidate = get_candidate()
                force = -1
                test_load = np.zeros([len(LHS_candidate), self.z_dim])
                for i in range(len(LHS_candidate)):
                    Fx = force * np.sin(test_load_z[i])
                    Fy = force * np.cos(test_load_z[i])
                    test_load[i, 2 * ((self.nely + 1) * test_load_x[i] + test_load_y[i] + 1) - 1] = Fy
                    test_load[i, 2 * ((self.nely + 1) * test_load_x[i] + test_load_y[i] + 1) - 2] = Fx
                return F_batch_candidate, test_load, random_candidate

            F_batch_candidate, test_load, random_candidate = get_test_load()

            # generate topology from test load.
            testing_num = 100
            phi_store_1 = []
            ratio = testing_num / self.batch_size
            final_error = 0
            for it in range(ratio):
                _, final_error_temp = self.sess.run([self.solver, self.recon_loss], feed_dict={
                    self.y_output: Y_test[it % ratio * self.batch_size:it % ratio * self.batch_size + self.batch_size].T,
                    self.F_input: test_load[it % ratio * self.batch_size:it % ratio * self.batch_size + self.batch_size]})
                final_error = final_error + final_error_temp
            final_error = final_error / testing_num * self.batch_size
            print('average testing error is: {}'.format(final_error))

            if len(index_ind) == terminate_step:
                saver.save(self.sess, 'Final_2D_2D/model_3D_final')
                print('Exiting. Saving the final model.....')
                break
            error_progress.append(final_error)

            phi_store = []
            ratio = testing_num / self.batch_size
            for it in range(ratio):
                phi_update = self.sess.run(self.phi_true, feed_dict={
                    self.F_input: F_batch_candidate[it % ratio * self.batch_size:it % ratio * self.batch_size + self.batch_size]})
                phi_store.append(phi_update)

            if not os.path.exists(self.directory_result):
                os.makedirs(self.directory_result)
            phi_gen = np.concatenate(phi_store, axis=1).T

            ########### Change for each run goes here
            Save_folder = 'Trial_100/'
            trial_num = 100  # can only contain numbers
            ###########
            if not os.path.exists(Save_folder): os.makedirs(Save_folder)
            sio.savemat('{}/phi_gen.mat'.format(Save_folder), {'phi_gen': phi_gen})
            sio.savemat('{}/random_candidate.mat'.format(Save_folder), {'random_candidate': random_candidate})
            sio.savemat('{}/error_progress.mat'.format(Save_folder), {'error_progress': error_progress})

            for it in range(ratio):
                phi_update_1 = self.sess.run(self.phi_true, feed_dict={
                    self.F_input: test_load[it % ratio * self.batch_size:it % ratio * self.batch_size + self.batch_size]})
                phi_store_1.append(phi_update_1)
            if not os.path.exists(self.directory_result): os.makedirs(self.directory_result)
            phi_gen_1 = np.concatenate(phi_store_1, axis=1).T
            sio.savemat('{}/phi_gen_test_input.mat'.format(Save_folder), {'phi_gen_test_input': phi_gen_1})

            if len(index_ind) % 50 == 0:
                sio.savemat('{}/phi_gen_test_input_{}.mat'.format(Save_folder, len(index_ind)),
                            {'phi_gen_test_input': phi_gen_1})
                eng = matlab.engine.start_matlab()
                eng.c_calculator(('{}/phi_gen_test_input_{}.mat'.format(Save_folder, str(len(index_ind)))),
                                 trial_num, len(index_ind), nargout=0)

            if self.Prepared_training_sample == False:
                budget = np.sum(sio.loadmat('{}/budget_store.mat')['budget_store'].reshape([-1])) + budget + 100

            # solve the worst one
            if self.Prepared_training_sample == False:
                eng = matlab.engine.start_matlab()
                eng.infill_high_dim(0, nargout=0)

            # evaluate the random samples and pick the worst one
            eng = matlab.engine.start_matlab()
            eng.cal_c_high_dim(nargout=0)


if __name__ == '__main__':
    # sess.close()/
    generator = TO_generator()
    generator.init_train()
    generator.build_model()
    generator.train_model()
