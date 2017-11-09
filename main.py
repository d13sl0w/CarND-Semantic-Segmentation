# coding: utf-8


import os.path
import numpy as np
import tensorflow as tf
import time
import helper
import cv2
import sys
from pprint import pprint
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import matplotlib.image as img


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))


# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16' 
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
        
    vgg_input_tensor = sess.graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob_tensor = sess.graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out_tensor = sess.graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out_tensor = sess.graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out_tensor = sess.graph.get_tensor_by_name('layer7_out:0')
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor,             vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    with tf.variable_scope('decoder_layer_0'):
        # 1x1 conv - resample for our class number in depth
        resampled_vgg_7 = tf.layers.conv2d(inputs=vgg_layer7_out, 
                                            filters=num_classes,
                                            kernel_size=(1, 1),
                                            padding='same',
                                            name='resampled_vgg_7',
                                            kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
        
        
    with tf.variable_scope('decoder_layer_1'):
        # trans-conv - upsample with fractionally strided conv
        upsampled_vgg_7_to_4 = tf.layers.conv2d_transpose(inputs=resampled_vgg_7,
                                                          filters=num_classes,
                                                          kernel_size=(4, 4),
                                                          strides=(2,2),
                                                          padding='same',
                                                          name='upsampled_vgg_7',
                                                          kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
        
        # 1x1 conv - resample for our class number in depth
        resampled_vgg_4 = tf.layers.conv2d(inputs=vgg_layer4_out, 
                                            filters=num_classes,
                                            kernel_size=(1, 1),
                                            padding='same',
                                            name='resampled_vgg_4',
                                            kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
        
        # skip connection - combine with mid-granularity encoding layer
        decoder_layer_1 = tf.add(upsampled_vgg_7_to_4, resampled_vgg_4, name='skip_connect_layer_0')
        

    with tf.variable_scope('decoder_layer_2'):
        # trans-conv - upsample with fractionally strided conv
        upsampled_vgg_4_to_3 = tf.layers.conv2d_transpose(inputs=decoder_layer_1,
                                                          filters=num_classes,
                                                          kernel_size=(4, 4),
                                                          strides=(2,2),
                                                          padding='same',
                                                          name='upsampled_vgg_4',
                                                          kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

        
        # 1x1 conv - resample for our class number in depth
        resampled_vgg_3 = tf.layers.conv2d(inputs=vgg_layer3_out, 
                                            filters=num_classes,
                                            kernel_size=(1, 1),
                                            padding='same',
                                            name='resampled_vgg_3',
                                            kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

        
        # skip connection - combine with mid-granularity encoding layer
        decoder_layer_2 = tf.add(upsampled_vgg_4_to_3, resampled_vgg_3, name='skip_connect_layer_1')
        

    DOUBLINGS_REMAINING = 4
    with tf.variable_scope('decoder_layer_3'):
        # FINAL UPSAMPLING TO ORIGINAL DIMENSIONS. WHAT IS THE IMPACT OF DOING A 
        #   MORE AGGRESSIVE UPSAMPLING VS. A SEQUENCE OF SMALLER ONES?
        # *** TODO: CREATE FCN LAYER GENERATOR (AT LEAST FOR NON-SKIP CONNECTION 
        #       LAYERS BASED ON TESTED TENSOR SHAPES FOUND IN RUN() BELOW!***)
        # Combinatorially uses different VGG, or whatever, layers to create skip
        #   layers and evaluates the model?
        upsampled_vgg_3_to_output = tf.layers.conv2d_transpose(inputs=decoder_layer_2,
                                                          filters=num_classes,
                                                          kernel_size=(4*DOUBLINGS_REMAINING, 
                                                                       4*DOUBLINGS_REMAINING),
                                                          strides=(2*DOUBLINGS_REMAINING,
                                                                   2*DOUBLINGS_REMAINING),
                                                          padding='same',
                                                          name='upsampled_final_output',
                                                          kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
        
    return upsampled_vgg_3_to_output
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    with tf.variable_scope('logits'):
        logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    
    with tf.variable_scope('target_reshape'):
        reshaped_label = tf.reshape(correct_label, (-1, num_classes))
    
#     with tf.variable_scope('cross_entropy'): # BREAKS HELPER FUNCTION FOR IMAGE INFERENCE SAMPLES
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=reshaped_label, logits=logits)#, name='softmax_xentropy_with_logits')
    mean_xentropy_loss = tf.reduce_mean(xentropy, name='mean_xentropy_loss')
    tf.summary.scalar('mean_xentropy_loss', mean_xentropy_loss)
    
    with tf.variable_scope('optimize'):
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(mean_xentropy_loss)#, global_step=global_step)
    return logits, train_op, mean_xentropy_loss 
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate 
    """    
    card_X = 0
    sum_loss = 0.

    for epoch in range(epochs):
        for X_batch, y_batch in get_batches_fn(batch_size):
            card_X += len(X_batch)
            feed_dict = {
                input_image: np.float32(X_batch),
                correct_label: y_batch,
                keep_prob: 0.8
            }
            _, loss_val = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            sum_loss += loss_val
        # this is technically right here but wrong later on. divide by batch count but last batch may not be fullsized
        batches_count = np.ceil(card_X / batch_size)
        mean_loss = sum_loss / batches_count
        print('mean_loss: {}'.format(mean_loss))
tests.test_train_nn(train_nn)

# tests I believe are locking out printing in jupyter from TF
# time.sleep(25)
# sys.stdout = sys.__stdout__

print('good')


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 15
    batch_size = 20
    # download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    tf.reset_default_graph()
    with tf.Session() as sess:
    # create necessary constants
        learning_rate = tf.placeholder(tf.float32)
        
    # Create model with given path for pretrained vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        vgg_tag = 'vgg16' 
        vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor,             vgg_layer4_out_tensor, vgg_layer7_out_tensor = load_vgg(sess, vgg_path)
        
    # TODO: Build NN using load_vgg, layers, and optimize function
        fcn_output = layers(vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        logits, train_op, mean_xentropy_loss = optimize(fcn_output, correct_label, learning_rate, num_classes)
        
    # run variable initializer
        sess.run(tf.global_variables_initializer())

                
    # Get train set helper and run training of NN
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, mean_xentropy_loss, vgg_input_tensor,
             correct_label, vgg_keep_prob_tensor, learning_rate)
            
#     # Get test set helper and run test of IOU and/or accuracy against final test set
#         get_test_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # create and write sample inference overlay images
        print("thiswhereiam")
        overlays = helper.gen_test_output(sess, logits, vgg_keep_prob_tensor, vgg_input_tensor,
                                          os.path.join(data_dir, 'data_road/testing'), image_shape)
        overlays = list(overlays)
        print(len(overlays))
        sample_folder = 'sample_inference_images/'
        directory = os.path.join(data_dir, sample_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        for file, image in overlays:
            img.imsave(os.path.join(directory, file), image)

#     # save the model
        model_folder = 'models/'
        directory = os.path.join(data_dir, model_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        saver = tf.train.Saver()    
        saver.save(sess, os.path.join(directory, 'working_beta_1.0'))

        
if __name__ == '__main__':
    run()

