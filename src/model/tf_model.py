import tensorflow as tf
import numpy as np

from src.data import utils
class Generator:
    def __init__(self):
        self.LEYE_Y, self.LEYE_X = 40, 42
        self.REYE_Y, self.REYE_X = 40, 86
        self.NOSE_Y, self.NOSE_X = 71, 64
        self.MOUTH_Y, self.MOUTH_X = 86, 64
        self.EYE_H, self.EYE_W = 40, 40
        self.MOUTH_H, self.MOUTH_W = 48,32
        self.NOSE_H, self.NOSE_W = 40,32

    def residual(self,x,filters, kernel, name):
        with tf.variable_scope('residual_'+name):
            conv0_res = tf.layers.conv2d(x,filters, kernel, (1,1),padding='same', activation=tf.nn.leaky_relu)
            conv1_res = tf.layers.conv2d(conv0_res,filters, kernel, (1,1), padding='same')
            conv_add_res = tf.add(conv1_res, x)
            residual_output = tf.nn.leaky_relu(conv_add_res)
            return residual_output


    def local(self,x,w,h,name):

        with tf.variable_scope("local_"+name):

            conv0 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     activation=tf.nn.leaky_relu)
            conv0 = tf.layers.batch_normalization(conv0)

            conv0 = self.residual(conv0, 64, (3,3), name+"local_conv0r")

            conv1 = tf.layers.conv2d(inputs=conv0, filters=128, kernel_size=(3,3), strides=(2, 2), padding='same',
                                     activation=tf.nn.leaky_relu)
            conv1 = tf.layers.batch_normalization(conv1)
            conv1 = self.residual(conv1, 128, (3,3), name+"local_conv1r")

            conv2 = tf.layers.conv2d(inputs=conv1, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                     activation=tf.nn.leaky_relu)
            conv2 = tf.layers.batch_normalization(conv2)
            conv2 = self.residual(conv2, 256, (3, 3), name+"local_conv2r")

            conv3 = tf.layers.conv2d(conv2, filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                     activation=tf.nn.leaky_relu)
            conv3 = tf.layers.batch_normalization(conv3)
            conv3 = self.residual(conv3, 512, (3, 3), "local_conv3r")

            deconv0 = tf.layers.conv2d_transpose(conv3, filters=256, kernel_size=(3, 3),strides=(2, 2),
                                                 padding='same', activation=tf.nn.relu)
            deconv1 = tf.layers.conv2d_transpose(tf.concat([deconv0, conv2], axis=-1), filters=128, kernel_size=(3, 3),
                                                 strides=(2, 2),padding='same', activation=tf.nn.relu)

            deconv2 = tf.layers.conv2d_transpose(tf.concat([deconv1, conv1], axis=-1), filters=64,kernel_size=(3, 3),
                                                 strides=(2, 2),padding='same', activation=tf.nn.relu)

            conv4 = tf.layers.conv2d(tf.concat([deconv2, conv0], axis=-1), filters=64,kernel_size=(3, 3),strides=(1,1),
                                     padding='same', activation=tf.nn.relu)

            conv5 = tf.layers.conv2d(conv4,filters=3,kernel_size=(3,3),strides=(1,1),padding='same',activation=tf.nn.relu)
            return conv4, conv5

    def combine_facial_parts(self,size_hw, leye, reye, nose, mouth):
        """
         combine the parts of facial_landmarks
        :param size_hw:
        :param leye:
        :param reye:
        :param nose:
        :param mouth:
        :return:
        """
        img_h, img_w = size_hw

        leye_img = tf.pad(leye,paddings=[[0,0],[int(self.LEYE_Y - self.EYE_H / 2), img_h - int(self.LEYE_Y + self.EYE_H / 2)],
            [int(self.LEYE_X - self.EYE_W / 2), img_w - int(self.LEYE_X + self.EYE_W / 2)],[0,0]],mode="CONSTANT", name=None,constant_values=0)

        reye_img = tf.pad(reye,paddings=[[0,0],[int(self.REYE_Y - self.EYE_H / 2), img_h - int(self.REYE_Y + self.EYE_H / 2)],
            [int(self.REYE_X - self.EYE_W / 2), img_w - int(self.REYE_X + self.EYE_W / 2)],[0,0]],mode="CONSTANT", name=None,constant_values=0)

        nose_img = tf.pad(nose,paddings=[[0,0],[int(self.NOSE_Y - self.NOSE_H / 2), img_h - int(self.NOSE_Y + self.NOSE_H / 2)],
            [int(self.NOSE_X - self.NOSE_W / 2), img_w - int(self.NOSE_X + self.NOSE_W / 2)],[0,0]],mode="CONSTANT", name=None,constant_values=0)

        mouth_img =tf.pad(mouth,[[0,0],[int(self.MOUTH_Y - self.MOUTH_H / 2),img_h - int(self.MOUTH_Y + self.MOUTH_H / 2)],
            [int(self.MOUTH_X - self.MOUTH_W / 2),img_w - int(self.MOUTH_X + self.MOUTH_W / 2)],[0,0]],mode="CONSTANT", name=None,constant_values=0)

        max1 =  tf.maximum(leye_img, reye_img)
        max2 = tf.maximum(max1,nose_img)
        return tf.maximum(max2,mouth_img)


    def forward(self,x,z,leye,reye,nose,mouth):

        with tf.variable_scope('generator'):
            mc_in_img128 = tf.image.resize_bilinear(x,[128,128])

            mc_in_img64 = tf.image.resize_bilinear(x,[64,64])
            mc_in_img32 = tf.image.resize_bilinear(x,[32,32])
            conv0 = tf.layers.conv2d(inputs=x, filters = 64,kernel_size=(7,7),strides =(1,1),padding='same',activation=tf.nn.leaky_relu)
            conv0 = tf.layers.batch_normalization(conv0)

            conv0 = self.residual(conv0,64, (7,7),"conv0r")

            conv1 = tf.layers.conv2d(inputs=conv0, filters=64, kernel_size=(5,5), strides=(2,2),padding='same', activation=tf.nn.leaky_relu)
            conv1 = tf.layers.batch_normalization(conv1)
            conv1 = self.residual(conv1, 64, (5,5),"conv1r")

            conv2 = tf.layers.conv2d(inputs=conv1,filters=128, kernel_size=(3,3), strides=(2, 2), padding='same',activation=tf.nn.leaky_relu)
            conv2 = tf.layers.batch_normalization(conv2)
            conv2 = self.residual(conv2, 128, (3, 3), "conv2r")

            conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3,3), strides=(2, 2),padding='same', activation=tf.nn.leaky_relu)
            conv3 = tf.layers.batch_normalization(conv3)
            conv3 = self.residual(conv3, 256, (3, 3), "conv3r")

            conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3,3), strides=(2,2),padding='same', activation=tf.nn.leaky_relu)
            conv4 = tf.layers.batch_normalization(conv4)
            conv4 = self.residual(conv4, 512, (3,3), "conv4r")
            conv4 = self.residual(conv4, 512, (3,3), "conv4r1")
            conv4 = self.residual(conv4, 512, (3,3), "conv4r2")
            conv4 = self.residual(conv4, 512, (3,3), "conv4r3")

            flatten = tf.layers.flatten(conv4)
            fc1 = tf.layers.dense(flatten, 512)
            fc2 = tf.layers.dense(fc1, 256)
            concat_noise = tf.concat([fc2,z],axis=1)
            fc3 = tf.layers.dense(concat_noise, 4096)

            fc3_reshaped = tf.reshape(fc3,[-1,1,1,4096])

            feat8 = tf.layers.conv2d_transpose(fc3_reshaped, filters=64,kernel_size=(8,8),strides=(1,1),padding ='valid', activation=tf.nn.relu)
            feat32 = tf.layers.conv2d_transpose(feat8, filters=32,kernel_size=(3,3),strides=(4,4), padding ='valid', activation=tf.nn.relu)
            feat64 = tf.layers.conv2d_transpose(feat32, filters=16, kernel_size=(3,3), strides=(2,2), padding='same', activation=tf.nn.relu)
            feat128 = tf.layers.conv2d_transpose(feat64, filters=8, kernel_size=(3,3), strides=(2,2), padding='same', activation=tf.nn.relu)


            deconv0 = tf.layers.conv2d_transpose(tf.concat([feat8,conv4],axis=-1), filters=512, kernel_size=(3,3), strides=(2,2),
                                                 padding='same', activation=tf.nn.relu)
            deconv1 = tf.layers.conv2d_transpose(tf.concat([deconv0,conv3],axis=-1),filters=256, kernel_size=(3,3), strides=(2,2),
                                                 padding='same', activation=tf.nn.relu)

            deconv2 = tf.layers.conv2d_transpose(tf.concat([deconv1,feat32, conv2,mc_in_img32], axis=-1), filters=128, kernel_size=(3, 3),
                                                 strides=(2, 2),
                                                 padding='same', activation=tf.nn.relu)
            deconv3 = tf.layers.conv2d_transpose(tf.concat([deconv2, conv1,mc_in_img64], axis=-1), filters=64, kernel_size=(3, 3),
                                                 strides=(2, 2),
                                                 padding='same', activation=tf.nn.relu)

            features_leye,front_leye = self.local(leye,40,40,'leye')
            features_reye,front_reye = self.local(reye,40,40,'reye')
            features_nose,front_nose = self.local(nose,40,32,'nose')
            features_mouth,front_mouth = self.local(mouth,48,32,'mouth')

            features_combined = self.combine_facial_parts([128,128],features_leye, features_reye, features_nose,features_mouth)

            conv5 = tf.layers.conv2d(tf.concat([feat128,conv0,features_combined,mc_in_img128,deconv3],axis=-1),filters=64,
                            kernel_size=(5,5), strides=(1,1), padding='same', activation=tf.nn.leaky_relu)

            conv6 = tf.layers.conv2d(conv5, filters=32, kernel_size=(3,3), strides=(1,1), padding='same',activation=tf.nn.leaky_relu)
            conv7 = tf.layers.conv2d(conv6, filters=3, kernel_size=(3,3), strides=(1,1), padding='same',activation=tf.nn.leaky_relu)
            return conv7

class Discriminator:
    def __init__(self):
        pass

    def forward(self,x):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3,3), strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3,3), strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3,3), strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3,3), strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            flatten = tf.layers.flatten(conv5)
            logits = tf.layers.dense(flatten, 1)
            output = tf.nn.sigmoid(logits)
            return logits,output

x,y,img_inp, leye_inp, reye_inp, nose_inp, mouth_inp, in_front_img, in_noise = utils.create_dataset()
batch_size = 4
gen = Generator()
dis = Discriminator()
x_input = tf.placeholder(dtype = tf.float32, shape=([None,128,128,3]))
z_input = tf.placeholder(dtype = tf.float32, shape=([None,100]))

x_leye = tf.placeholder(dtype = tf.float32, shape = [None,40,40,3])
x_reye = tf.placeholder(dtype = tf.float32, shape = [None,40,40,3])
x_nose = tf.placeholder(dtype = tf.float32, shape = [None,40,32,3])
x_mouth = tf.placeholder(dtype = tf.float32, shape = [None,48,32,3])
y_fake = tf.placeholder(dtype = tf.float32, shape = [None,1])
y_real = tf.placeholder(dtype = tf.float32, shape = [None,1])

fake_labels = y[batch_size:]
real_labels = y[:batch_size]

noise = np.random.random([4,100])
Gen_out_op = gen.forward(x_input,z_input,x_leye,x_reye,x_nose,x_mouth)
Dis_fake_out_op, fake_output = dis.forward(Gen_out_op)
Dis_real_out_op, real_output = dis.forward(x_input)

# D_real_loss = tf.multiply(y_real, tf.log(real_output)) + tf.multiply(tf.subtract(1.0,y_real), tf.log(tf.subtract(1.0, real_output)))
# print D_real_loss
D_real_loss = tf.losses.sigmoid_cross_entropy(real_output, y_real)
# print D_real_loss
# D_fake_loss = tf.multiply(y_fake, tf.log(fake_output)) + tf.multiply(tf.subtract(1.0,y_fake), tf.log(tf.subtract(1.0, fake_output)))

D_fake_loss = tf.losses.sigmoid_cross_entropy(fake_output,y_fake)
# G_loss = tf.multiply(y_real, tf.log(fake_output)) + tf.multiply(tf.subtract(1.0,y_real), tf.log(tf.subtract(1.0, fake_output)))

D_loss = tf.add(D_real_loss, D_fake_loss)
G_loss = tf.losses.sigmoid_cross_entropy(fake_output,y_real)

D_optimizer = tf.train.AdamOptimizer(0.00001).minimize(D_loss)
G_optimizer = tf.train.AdamOptimizer(0.00001).minimize(G_loss)
avg_gen_loss, avg_dis_loss=0,0

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for i in range(100):
        Dis_loss, _, out1, out2 = sess.run([D_loss,D_optimizer,fake_output,real_output],feed_dict={x_input:img_inp, z_input:noise, x_leye:leye_inp, x_reye:reye_inp,x_nose:nose_inp,
                                                               x_mouth:mouth_inp, y_fake:fake_labels, y_real:real_labels})
        Gen_loss, _ = sess.run([G_loss,G_optimizer],feed_dict={x_input:img_inp, z_input:noise, x_leye:leye_inp, x_reye:reye_inp,x_nose:nose_inp,
                                                               x_mouth:mouth_inp, y_fake:fake_labels, y_real:real_labels})

        avg_gen_loss = np.average(Gen_loss)
        avg_dis_loss = np.average(Dis_loss)
        # print avg_dis_loss, avg_gen_loss
        #
        print "the loss for discriminator in epoch {} is {}".format(i,Dis_loss)
        print "the loss for generator in epoch {} is {}".format(i,Gen_loss)
