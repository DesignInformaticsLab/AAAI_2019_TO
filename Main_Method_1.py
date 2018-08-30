import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import random
import matlab.engine

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def P(z):
    h1 = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    h2_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(tf.reshape(h1,[batch_size, width/8, height/8, 1]),
                                                  deconv2_1_weight, strides=[1, 1, 1, 1], padding='SAME',
                                       output_shape=[batch_size, width/8, height/8, deconv2_1_features]),deconv2_1_bias))

    h2_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h2_1,deconv2_2_weight, strides=[1, 2, 2, 1], padding='SAME',
                                       output_shape=[batch_size, width/4, height/4, deconv2_2_features]),deconv2_2_bias))

    h3_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h2_2, deconv3_1_weight, strides=[1, 1, 1, 1], padding='SAME',
                                       output_shape=[batch_size, width/4, height/4, deconv3_1_features]),deconv3_1_bias))

    h3_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h3_1, deconv3_2_weight, strides=[1, 2, 2, 1], padding='SAME',
                                       output_shape=[batch_size, width/2, height/2, deconv3_2_features]),deconv3_2_bias))

    h4_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h3_2, deconv4_1_weight, strides=[1, 1, 1, 1], padding='SAME',
                                       output_shape=[batch_size, width/2, height/2, deconv4_1_features]),deconv4_1_bias))

    h4_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h4_1, deconv4_2_weight, strides=[1, 2, 2, 1], padding='SAME',
                                       output_shape=[batch_size, width/1, height/1, deconv4_2_features]),deconv4_2_bias))

    h5 = (tf.add(tf.nn.conv2d_transpose(h4_2, deconv5_weight, strides=[1, 1, 1, 1], padding='SAME',
                                        output_shape=[batch_size, width/1, height/1, 1]),deconv5_bias))

    prob = tf.nn.sigmoid(h5)
#     prob = 1 / (1 + tf.exp(-h5))

    return prob


# Input parameter
nelx, nely = 12*10, 4*10
nn = nelx*nely
batch_size=10
initial_num=100
Prepared_training_sample = True # True if samples are pre-solved offline

# network parameter
z_dim = 2*41*41
width = nely
height = nelx
h_dim = width/8*height/8


deconv2_1_features=32*3
deconv2_2_features=32*3
deconv3_1_features=32*2
deconv3_2_features=32*2
deconv4_1_features=32
deconv4_2_features=32



sess=tf.Session()
saver = tf.train.import_meta_graph('./Final_1D/model_1D_final.meta')
saver.restore(sess, tf.train.latest_checkpoint('./Final_1D/'))
graph = tf.get_default_graph()

F_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))


P_W1_direction = sess.run('P_W1:0')

P_W1 = tf.Variable(P_W1_direction, dtype=tf.float32)

#print(P_W)


P_b1 =  sess.run('P_b1:0')
P_b1 = tf.Variable(P_b1, dtype = tf.float32)

deconv2_1_weight = tf.Variable(sess.run('deconv2_1_weight:0'),dtype = tf.float32)
deconv2_1_bias = tf.Variable(sess.run('deconv2_1_bias:0'),dtype = tf.float32)
deconv2_2_weight = tf.Variable(sess.run('deconv2_2_weight:0'),dtype = tf.float32)
deconv2_2_bias = tf.Variable(sess.run('deconv2_2_bias:0'),dtype = tf.float32)
deconv3_1_weight = tf.Variable(sess.run('deconv3_1_weight:0'),dtype = tf.float32)
deconv3_1_bias = tf.Variable(sess.run('deconv3_1_bias:0'),dtype = tf.float32)
deconv3_2_weight = tf.Variable(sess.run('deconv3_2_weight:0'),dtype = tf.float32)
deconv3_2_bias = tf.Variable(sess.run('deconv3_2_bias:0'),dtype = tf.float32)
deconv4_1_weight = tf.Variable(sess.run('deconv4_1_weight:0'),dtype = tf.float32)
deconv4_1_bias = tf.Variable(sess.run('deconv4_1_bias:0'),dtype = tf.float32)
deconv4_2_weight = tf.Variable(sess.run('deconv4_2_weight:0'),dtype = tf.float32)
deconv4_2_bias = tf.Variable(sess.run('deconv4_2_bias:0'),dtype = tf.float32)
deconv5_weight = tf.Variable(sess.run('deconv5_weight:0'),dtype = tf.float32)
deconv5_bias = tf.Variable(sess.run('deconv5_bias:0'),dtype = tf.float32)



# deconv2_1_weight = tf.Variable(tf.truncated_normal([4, 4, deconv2_1_features, 1],
#                                                stddev=0.1, dtype=tf.float32))
# deconv2_1_bias = tf.Variable(tf.zeros([deconv2_1_features], dtype=tf.float32))
#
# deconv2_2_weight = tf.Variable(tf.truncated_normal([4, 4, deconv2_2_features,deconv2_1_features],
#                                                stddev=0.1, dtype=tf.float32))
# deconv2_2_bias = tf.Variable(tf.zeros([deconv2_2_features], dtype=tf.float32))
#
# deconv3_1_weight = tf.Variable(tf.truncated_normal([4, 4, deconv3_1_features, deconv2_2_features],
#                                                stddev=0.1, dtype=tf.float32))
# deconv3_1_bias = tf.Variable(tf.zeros([deconv3_1_features], dtype=tf.float32))
#
# deconv3_2_weight = tf.Variable(tf.truncated_normal([4, 4, deconv3_2_features, deconv3_1_features],
#                                                stddev=0.1, dtype=tf.float32))
# deconv3_2_bias = tf.Variable(tf.zeros([deconv3_2_features], dtype=tf.float32))
#
# deconv4_1_weight = tf.Variable(tf.truncated_normal([4, 4, deconv4_1_features, deconv3_2_features],
#                                                stddev=0.1, dtype=tf.float32))
# deconv4_1_bias = tf.Variable(tf.zeros([deconv4_1_features], dtype=tf.float32))
#
# deconv4_2_weight = tf.Variable(tf.truncated_normal([8, 8, deconv4_2_features, deconv4_1_features],
#                                                stddev=0.1, dtype=tf.float32))
# deconv4_2_bias = tf.Variable(tf.zeros([deconv4_2_features], dtype=tf.float32))
#
# deconv5_weight = tf.Variable(tf.truncated_normal([8, 8, 1, deconv4_2_features],
#                                                stddev=0.1, dtype=tf.float32))
# deconv5_bias = tf.Variable(tf.zeros([1], dtype=tf.float32))


P_output = P(F_input)

phi_true = tf.transpose(tf.reshape(P_output,[batch_size,nn]))


global_step=tf.Variable(0,trainable=False)
starter_learning_rate=0.005
learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,1000,0.98,staircase=True)
y_output=tf.placeholder(tf.float32, shape=([nn, batch_size]))
recon_loss = tf.reduce_sum((phi_true-y_output)**2)/batch_size
if 0:
    solver = tf.train.AdamOptimizer(learning_rate).minimize(recon_loss,global_step)
else:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    vars = [P_W1, P_b1,
            deconv2_1_weight, deconv2_1_bias,
            deconv2_2_weight, deconv2_2_bias,
            deconv3_1_weight, deconv3_1_bias,
            deconv3_2_weight, deconv3_2_bias,
            deconv4_1_weight, deconv4_1_bias,
            deconv4_2_weight, deconv4_2_bias,
#            deconv5_weight, deconv5_bias,
            ]
    grads_g = optimizer.compute_gradients(recon_loss, var_list=vars)
    solver = optimizer.apply_gradients(grads_g,global_step)

sess.run(tf.global_variables_initializer())
# generating initial points
directory_data='experiment_data'
directory_model='model_save'
directory_result='experiment_result'
directory_model_3D = 'model_save_3D'
directory_model_3D_final = 'Final_3D'


LHS = sio.loadmat('{}/LHS_train.mat'.format(directory_data))['LHS_train'] # pre-sampling the loading condition offline

LHS[:,0] = LHS[:,0]-81
LHS[:,1] = LHS[:,1]-1

LHS_x=np.int32(LHS[:,0])
LHS_y=np.int32(LHS[:,1])
LHS_z=LHS[:,2]

force=-1
F_batch = np.zeros([len(LHS), z_dim])
error_store=[]
for i in range(len(LHS)):
    Fx = force * np.sin(LHS_z[i])
    Fy = force * np.cos(LHS_z[i])
    F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
    F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx


index_ind = random.sample(range(0,len(LHS)),initial_num) # initial start with 100, can be modified

if Prepared_training_sample==True:
    pass
else:
    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    sio.savemat('{}/index_ind.mat'.format(directory_result),{'index_ind':index_ind})
    eng = matlab.engine.start_matlab()
    eng.infill_high_dim(1,nargout=0)

Y_test = sio.loadmat('{}/phi_true_test2.mat'.format(directory_data))['phi_true_test'] # prepared off-line
test_load = sio.loadmat('{}/LHS_test2.mat'.format(directory_data))['LHS_test']  # prepared off-line
test_load[:,0] = test_load[:,0]-81
test_load[:,1] = test_load[:,1]-1

test_load_x=np.int32(test_load[:,0])
test_load_y=np.int32(test_load[:,1])
test_load_z=test_load[:,2]

budget=0
error_progress = []
final_error=float('inf')
terminate_criteria=10
terminate_step = 451
starting_loss = 100
decay_rate = 0.6
# one-shot algorithm
while len(index_ind) <= terminate_step:
    print("requirement doesn't match, current final_error={}, keep sampling".format(final_error))
    try:
        add_point_index=sio.loadmat('{}/add_point_index.mat'.format(directory_result))['add_point_index'][0]
        index_ind=list(add_point_index)+index_ind
    except:
        print("add_point_index bug")
        pass
    loss_list = []
    global_step_loss = len(index_ind)-initial_num
    decay_loss = starting_loss * decay_rate**global_step_loss
    Y_train = sio.loadmat('{}/phi_true_train.mat'.format(directory_data))['phi_true_train']
    F_batch = np.zeros([len(LHS), z_dim])
    for i in range(len(LHS)):
        F_batch[i,0]=LHS_x[i]
        F_batch[i,1]=LHS_y[i]
        F_batch[i,2]=LHS_z[i]
    force=-1
    F_batch = np.zeros([len(LHS), z_dim])
    for i in range(len(LHS)):
        Fx = force * np.sin(LHS_z[i])
        Fy = force * np.cos(LHS_z[i])
        F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
        F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx

    for it in range(100000):
        random_ind=np.random.choice(index_ind,batch_size,replace=False)

        _,error=sess.run([solver, recon_loss],feed_dict={y_output:Y_train[random_ind].T,F_input:F_batch[random_ind]})
        if len(index_ind)%100 == 0: # plot loss curve
            loss_list.append(error)
        if it%100 == 0:
            print('iteration:{}, recon_loss:{}, number of data used is:{}'.format(it,error,(len(index_ind))))
        #if error <= 1 : # try exponential decay
        if error <= decay_loss or error<=1:
            print('loss threshold is: {}'.format(decay_loss))
            if not os.path.exists(directory_model):
                os.makedirs(directory_model)
            saver=tf.train.Saver()
            saver.save(sess, '{}/model_sample_{}'.format(directory_model_3D,len(index_ind)))
            print('converges, saving the model.....')
            break
    print('number of data used is:{}'.format(len(index_ind)))

    candidate_pool=list(set(list(np.int32(np.linspace(0,len(Y_train)-1,len(Y_train)))))-set(index_ind))
    random_candidate=np.random.choice(candidate_pool, 100 ,replace=False)
    LHS_candidate = sio.loadmat('{}/LHS_train.mat'.format(directory_data))['LHS_train'][random_candidate]
    LHS_candidate[:,0] = LHS_candidate[:,0]-81
    LHS_candidate[:,1] = LHS_candidate[:,1]-1

    LHS_x_candidate=np.int32(LHS_candidate[:,0])
    LHS_y_candidate=np.int32(LHS_candidate[:,1])
    LHS_z_candidate=LHS_candidate[:,2]

    force=-1
    F_batch_candidate= np.zeros([len(LHS_candidate), z_dim])
    for i in range(len(LHS_candidate)):
        Fx = force * np.sin(LHS_z[i])
        Fy = force * np.cos(LHS_z[i])
        F_batch_candidate[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
        F_batch_candidate[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx


    testing_num = 100
    # generate topology from test load.
    rho_gen_1 = []
    phi_store_1 = []
    force=-1
    test_load= np.zeros([len(LHS_candidate), z_dim])
    for i in range(len(LHS_candidate)):
        Fx = force * np.sin(test_load_z[i])
        Fy = force * np.cos(test_load_z[i])
        test_load[i,2*((nely+1)*test_load_x[i]+test_load_y[i]+1)-1]=Fy
        test_load[i,2*((nely+1)*test_load_x[i]+test_load_y[i]+1)-2]=Fx
    ratio=testing_num/batch_size
    final_error=0
    for it in range(ratio):
        _,final_error_temp=sess.run([solver, recon_loss],feed_dict={y_output:Y_test[it%ratio*batch_size:it%ratio*batch_size+batch_size].T,
                                                                    F_input:test_load[it%ratio*batch_size:it%ratio*batch_size+batch_size]})
        final_error=final_error + final_error_temp
    final_error=final_error/testing_num * batch_size
    print('average testing error is: {}'.format(final_error))

    if len(index_ind) == terminate_step:
        saver.save(sess, 'Final_2D_2D/model_3D_final')
        print('Exiting. Saving the final model.....')
        break

    error_progress.append(final_error)

    rho_gen=[]
    phi_store=[]
    ratio=testing_num/batch_size
    for it in range(ratio):
        phi_update=sess.run(phi_true,feed_dict={F_input:F_batch_candidate[it%ratio*batch_size:it%ratio*batch_size+batch_size]})
        phi_store.append(phi_update)

    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    phi_gen=np.concatenate(phi_store,axis=1).T


    ########### Change for each run goes here
    Save_folder = 'Trial_102/'
    trial_num = 102 #can only contain numbers
    ###########


    if not os.path.exists(Save_folder):
        os.makedirs(Save_folder)
    sio.savemat('{}/phi_gen.mat'.format(Save_folder),{'phi_gen':phi_gen})
    sio.savemat('{}/random_candidate.mat'.format(Save_folder),{'random_candidate':random_candidate})
    sio.savemat('{}/error_progress.mat'.format(Save_folder),{'error_progress':error_progress})

    for it in range(ratio):
        phi_update_1 = sess.run(phi_true, feed_dict={
            F_input: test_load[it % ratio * batch_size:it % ratio * batch_size + batch_size]})
        phi_store_1.append(phi_update_1)

    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    phi_gen_1 = np.concatenate(phi_store_1, axis=1).T

    sio.savemat('{}/phi_gen_test_input.mat'.format(Save_folder), {'phi_gen_test_input': phi_gen_1})

    if len(index_ind) % 50 == 0:
        sio.savemat('{}/phi_gen_test_input_{}.mat'.format(Save_folder,len(index_ind)), {'phi_gen_test_input': phi_gen_1})
        eng = matlab.engine.start_matlab()
        eng.c_calculator(('{}/phi_gen_test_input_{}.mat'.format(Save_folder,str(len(index_ind)))),trial_num,len(index_ind),nargout=0)

    if Prepared_training_sample==False:
        budget=np.sum(sio.loadmat('{}/budget_store.mat')['budget_store'].reshape([-1]))+budget+100

    # solve the worst one
    if Prepared_training_sample == False:
        eng = matlab.engine.start_matlab()
        eng.infill_high_dim(0,nargout=0)

    # evaluate the random samples and pick the worst one
    eng = matlab.engine.start_matlab()
    eng.cal_c_high_dim(nargout=0)



sess.close()
