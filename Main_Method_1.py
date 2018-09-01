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
nelx, nely = 12*10, 4*10 # 120 * 40 grid
nn = nelx*nely # Total number of elements
batch_size=10 # Batch Size
initial_num=100 # Initial number of data points
Prepared_training_sample = True # True if samples are pre-solved offline

# network parameter
z_dim = 2*41*41 # Input dimension
width = nely
height = nelx
h_dim = width/8*height/8 # Nodes for the first full connected layer


deconv2_1_features=32*3
deconv2_2_features=32*3
deconv3_1_features=32*2
deconv3_2_features=32*2
deconv4_1_features=32
deconv4_2_features=32



sess=tf.Session()
saver = tf.train.import_meta_graph('./Final_1D/model_1D_final.meta') # Import 1D weight
saver.restore(sess, tf.train.latest_checkpoint('./Final_1D/')) # Import Checkpoint file
graph = tf.get_default_graph()

F_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim])) # Input Force


# Transfer weights

P_W1_direction = sess.run('P_W1:0')
P_W1 = tf.Variable(P_W1_direction, dtype=tf.float32)
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



## Vanilla weights
# P_W1 = tf.Variable(xavier_init([z_dim, h_dim]),name="P_W1")
# P_b1 = tf.Variable(tf.zeros(shape=[h_dim]),name="P_b1")
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

phi_true = tf.transpose(tf.reshape(P_output,[batch_size,nn])) # Phi gen


global_step=tf.Variable(0,trainable=False)
starter_learning_rate=0.005 # Learning Rate
learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,1000,0.98,staircase=True) # Exponential decay of learning rate
y_output=tf.placeholder(tf.float32, shape=([nn, batch_size])) # Ground Truth
recon_loss = tf.reduce_sum((phi_true-y_output)**2)/batch_size # Reduced sum of ground Truth minus Phi Gen squared over batch size (Recon loss per sample)
solver = tf.train.AdamOptimizer(learning_rate).minimize(recon_loss,global_step) # Solve for weights
sess.run(tf.global_variables_initializer()) # Initialize all weights
saver = tf.train.Saver() # Load Graph saver
# generating initial points
directory_data='experiment_data'
directory_model='model_save'
directory_result='experiment_result'
directory_model_3D = 'model_save_3D'
directory_model_3D_final = 'Final_3D'

LHS = sio.loadmat('{}/LHS_train.mat'.format(directory_data))['LHS_train'] # Training load input

# Data Pre-processing for training input
LHS[:,0] = LHS[:,0]-81
LHS[:,1] = LHS[:,1]-1

LHS_x=np.int32(LHS[:,0])
LHS_y=np.int32(LHS[:,1])
LHS_z=LHS[:,2]


if Prepared_training_sample==True:
    pass
else:
    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    sio.savemat('{}/index_ind.mat'.format(directory_result),{'index_ind':index_ind})
    eng = matlab.engine.start_matlab()
    eng.infill_high_dim(1,nargout=0)

Y_test = sio.loadmat('{}/phi_true_test2.mat'.format(directory_data))['phi_true_test'] # Load phi gen from Testing set
test_load = sio.loadmat('{}/LHS_test2.mat'.format(directory_data))['LHS_test']  # Load input from testing set
Y_train = sio.loadmat('{}/phi_true_train.mat'.format(directory_data))['phi_true_train']  # Load phi gen from training set

# Data Pre-processing for testing input
test_load[:,0] = test_load[:,0]-81
test_load[:,1] = test_load[:,1]-1
test_load_x=np.int32(test_load[:,0])
test_load_y=np.int32(test_load[:,1])
test_load_z=test_load[:,2]

budget=0
error_progress = [] # Testing Error plot
final_error=float('inf')
terminate_criteria=10 # Required Testing error
terminate_step = 451 # Required number of data
starting_loss = 100 # starting target training loss
decay_rate = 0.8 # Exponential decay rate for target training loss
index_ind = random.sample(range(0,len(LHS)),initial_num) # Randomly select index from all training data

########### Change for each run goes here
Save_folder = 'Trial_test_2/' # Where all phi gen from testing input for this trial is stored here
trial_num = 2  # can only contain numbers
###########

# one-shot algorithm
while len(index_ind) <= terminate_step:
    print("requirement doesn't match, current final_error={}, keep sampling".format(final_error))
    try:
        add_point_index=sio.loadmat('{}/add_point_index.mat'.format(directory_result))['add_point_index'][0] #Load index of the new data from Experimental results folder
        index_ind=list(add_point_index)+index_ind # add new index into data sets
    except:
        print("add_point_index bug") #Report a bug if there is no add point index
        pass
    global_step_loss = len(index_ind)-initial_num # Global step for target loss decay
    decay_loss = starting_loss * decay_rate**global_step_loss # Exponential decay for target training loss
    F_batch = np.zeros([len(LHS), z_dim]) #F_batch is the input loading data

    force=-1 # Total force

    for i in range(len(LHS)):
        Fx = force * np.sin(LHS_z[i]) # Force in x direction
        Fy = force * np.cos(LHS_z[i]) # Force in y direction
        F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy # Embed the force in the F_batch
        F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx

    for it in range(100000): # Max iteration allowed when training
        random_ind=np.random.choice(index_ind,batch_size,replace=False) # Randomly chose batch size number of the data from the data set

        _,error=sess.run([solver, recon_loss],feed_dict={y_output:Y_train[random_ind].T,F_input:F_batch[random_ind]}) # Solve for weight using Y_train: phi gen train; F_batch: training in put

        if it%100 == 0:
            print('iteration:{}, recon_loss:{}, number of data used is:{}'.format(it,error,(len(index_ind)))) # Report training reconstruction error and number of data used per 100 iteration
        if error <= 1 :
        #if (error <= decay_loss and it >1000) or error<=5:
            if not os.path.exists(directory_model):
                os.makedirs(directory_model)
            print('converges, saving the model.....')
            break

    candidate_pool=list(set(list(np.int32(np.linspace(0,len(Y_train)-1,len(Y_train)))))-set(index_ind)) # Candidate pool is all the training data except for those we are using
    random_candidate=np.random.choice(candidate_pool, 100 ,replace=False) # Randomly pick 100 data from the candidate pool
    LHS_candidate = sio.loadmat('{}/LHS_train.mat'.format(directory_data))['LHS_train'][random_candidate] # Input load for random candidates from training set

    # Data pre-processing for LHS_candidate
    LHS_candidate[:,0] = LHS_candidate[:,0]-81
    LHS_candidate[:,1] = LHS_candidate[:,1]-1

    LHS_x_candidate=np.int32(LHS_candidate[:,0]) # Validation x
    LHS_y_candidate=np.int32(LHS_candidate[:,1]) # Validation y
    LHS_z_candidate=LHS_candidate[:,2] # Validation z

    # Transform LHS_candidate to 41*41*2 input (Training input)
    F_batch_candidate= np.zeros([len(LHS_candidate), z_dim])
    for i in range(len(LHS_candidate)):
        Fx = force * np.sin(LHS_z_candidate[i])
        Fy = force * np.cos(LHS_z_candidate[i])
        F_batch_candidate[i,2*((nely+1)*LHS_x_candidate[i]+LHS_y_candidate[i]+1)-1]=Fy
        F_batch_candidate[i,2*((nely+1)*LHS_x_candidate[i]+LHS_y_candidate[i]+1)-2]=Fx


    testing_num = 100
    # generate topology from test load.
    rho_gen_1 = []
    phi_store_1 = []

    # Transform test load to 41*41*2 input (Testing input)
    test_load= np.zeros([len(LHS_candidate), z_dim])
    for i in range(len(LHS_candidate)):
        Fx = force * np.sin(test_load_z[i])
        Fy = force * np.cos(test_load_z[i])
        test_load[i,2*((nely+1)*test_load_x[i]+test_load_y[i]+1)-1]=Fy
        test_load[i,2*((nely+1)*test_load_x[i]+test_load_y[i]+1)-2]=Fx


    ratio=testing_num/batch_size
    final_error=0
    for it in range(ratio):
        final_error_temp=sess.run(recon_loss,feed_dict={y_output:Y_test[it%ratio*batch_size:it%ratio*batch_size+batch_size].T,
                                                                    F_input:test_load[it%ratio*batch_size:it%ratio*batch_size+batch_size]}) #Get testing loss using testing input and output
        final_error=final_error + final_error_temp
    final_error=final_error/testing_num * batch_size # Averaged testing error for one single sample
    print('average testing error is: {}'.format(final_error))

    if len(index_ind) == terminate_step: # Save final model before terminate
        saver.save(sess, 'Final_2D_2D/model_3D_final')
        print('Exiting. Saving the final model.....')
        break

    error_progress.append(final_error) #  Testing Error plot

    rho_gen=[]
    phi_store=[]
    ratio=testing_num/batch_size
    for it in range(ratio):
        phi_update=sess.run(phi_true,feed_dict={F_input:F_batch_candidate[it%ratio*batch_size:it%ratio*batch_size+batch_size]}) #Get validation phi gen from training input
        phi_store.append(phi_update)

    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    phi_gen=np.concatenate(phi_store,axis=1).T #Phi gen for validation
    sio.savemat('{}/phi_gen.mat'.format(directory_result),{'phi_gen':phi_gen})
    sio.savemat('{}/random_candidate.mat'.format(directory_result),{'random_candidate':random_candidate})


    if not os.path.exists(Save_folder):
        os.makedirs(Save_folder)
    sio.savemat('{}/error_progress.mat'.format(Save_folder),{'error_progress':error_progress}) # Save testing error information

    for it in range(ratio):
        phi_update_1 = sess.run(phi_true, feed_dict={
            F_input: test_load[it % ratio * batch_size:it % ratio * batch_size + batch_size]})
        phi_store_1.append(phi_update_1)
    phi_gen_1 = np.concatenate(phi_store_1, axis=1).T # Phi gen from testing input

    sio.savemat('{}/phi_gen_test_input.mat'.format(Save_folder), {'phi_gen_test_input': phi_gen_1}) # Store phi gen test input in the trial folder

    if len(index_ind) % 100 == 0:
        sio.savemat('{}/phi_gen_test_input_{}.mat'.format(Save_folder,len(index_ind)), {'phi_gen_test_input': phi_gen_1}) # Save phi gen test input per certain number of data added
        eng = matlab.engine.start_matlab()
        eng.c_calculator(('{}/phi_gen_test_input_{}.mat'.format(Save_folder,str(len(index_ind)))),trial_num,len(index_ind),nargout=0) # Calculate the compliance for phi gen test input and save it

    if Prepared_training_sample==False:
        budget=np.sum(sio.loadmat('{}/budget_store.mat')['budget_store'].reshape([-1]))+budget+100

    # solve the worst one
    if Prepared_training_sample == False:
        eng = matlab.engine.start_matlab()
        eng.infill_high_dim(0,nargout=0)

    # evaluate the random samples and pick the worst one
    eng = matlab.engine.start_matlab()
    eng.cal_c_high_dim(nargout=0) # Proceed to cal_c_high.m



