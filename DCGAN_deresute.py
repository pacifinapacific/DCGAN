
import tensorflow as tf
import os

def batch(batch_size=64):
    paths=[]
    
    topdir=os.path.join('image','faces')
    #os.path.joinでpathの結合
    for dirpath,_,files in os.walk(topdir, followlinks=True): #os.walkでディレクトリ走査、dirpath,dirnames,filenamesのタプルを返す
        paths +=[os.path.join(dirpath,file) for file in files] #patsに取得したファイルのパスを追加していく
    queue=tf.train.slice_input_producer([paths])#元データをバッチで使用できる形式にする [N,image1,image2,,,]
    png=tf.read_file(queue[0])#画像を読み込み
    image=tf.image.decode_png(png, channels=3) #画像をtensorに変換,channel=3でRGB指定
    image=tf.image.resize_images(image, [64,64])#画像をリサイズ
    image=tf.subtract(tf.divide(image,127.5),1.0) #画像を-1~1で正規化
    return tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=len(paths)+3*batch_size,
        min_after_dequeue=len(paths)) #バッチ生成

def generator(inputs,batch_size):

    '''

    Args:
        inputs:[batch_size,100]のTensor 10は乱数zの数
        batch_size=128
    Return:
        生成結果の[batch_size,64,64,3]のTensor
    '''

    
    with tf.variable_scope('g'):
        with tf.variable_scope('reshape'):
            weight0=tf.get_variable(
                'w',[100,4*4*1024],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias0=tf.get_variable(
                'b',shape=[4*4*1024],
                initializer=tf.zeros_initializer())
            fc0=tf.add(tf.matmul(inputs,weight0),bias0)
            out_reshape=tf.reshape(fc0,[batch_size,4,4,1024])
            mean0,var0=tf.nn.moments(out_reshape,[0,1,2])
            out_norm0=tf.nn.batch_normalization(out_reshape,mean0,var0,None,None,1e-5)
            out0=tf.nn.relu(out_norm0)

        with tf.variable_scope('conv_transpose1'):
            weight1=tf.get_variable(
                'w',[5,5,512,1024],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1=tf.get_variable(
                'b',shape=[512],
                initializer=tf.zeros_initializer())
            deconv1=tf.nn.conv2d_transpose(out0,weight1,[batch_size,8,8,512],[1,2,2,1])
            out_add1=tf.add(deconv1,bias1)
            mean1,var1=tf.nn.moments(out_add1,[0,1,2])
            out_norm1=tf.nn.batch_normalization(out_add1,mean1,var1,None,None,1e-5)
            out1=tf.nn.relu(out_norm1)

        with tf.variable_scope('conv_transpose2'):
            weight2=tf.get_variable(
                'w',[5,5,256,512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias2=tf.get_variable(
                'b',shape=[256],
                initializer=tf.zeros_initializer())
            deconv2=tf.nn.conv2d_transpose(out1,weight2,[batch_size,16,16,256],[1,2,2,1])
            out_add2=tf.add(deconv2,bias2)
            mean2,var2=tf.nn.moments(out_add2,[0,1,2])
            out_norm2=tf.nn.batch_normalization(out_add2,mean2,var2,None,None,1e-5)
            out2=tf.nn.relu(out_norm2)
        
        with tf.variable_scope('conv_transpose3'):
            weight3=tf.get_variable(
                'w',[5,5,128,256],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias3=tf.get_variable(
                'b',shape=[128],
            initializer=tf.zeros_initializer())
            deconv3=tf.nn.conv2d_transpose(out2,weight3,[batch_size,32,32,128],[1,2,2,1])
            out_add3=tf.add(deconv3,bias3)
            mean3,var3=tf.nn.moments(out_add3,[0,1,2])
            out_norm3=tf.nn.batch_normalization(out_add3,mean3,var3,None,None,1e-5)
            out3=tf.nn.relu(out_norm3)

        with tf.variable_scope('conv_transpose4'):
            weight4=tf.get_variable(
                'w',[5,5,3,128],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias4=tf.get_variable(
                'b',shape=[3],
                initializer=tf.zeros_initializer())
            deconv4=tf.nn.conv2d_transpose(out3,weight4,[batch_size,64,64,3],[1,2,2,1])
            out4=tf.nn.tanh(tf.add(deconv4,bias4))

    return out4


def disctiminator(inputs,reuse=False):

    '''

    Args:
        inputs: [batch_size,height(=32),width(=32),channels(=1)]のTensor
        reuse 変数を再利用するか否か
    Returns：
        推論結果の[bathc_size,2]のTensor
    '''

    with tf.variable_scope('d'):

        with tf.variable_scope('conv1',reuse=reuse):
            weight1=tf.get_variable(
                'w',[5,5,3,64],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1=tf.get_variable(
                'b',shape=[64],
                initializer=tf.zeros_initializer())
            conv1=tf.nn.conv2d(inputs,weight1,[1,2,2,1],'SAME')
            out1=tf.nn.leaky_relu(tf.add(conv1,bias1))

        with tf.variable_scope('conv2',reuse=reuse):
            weight2=tf.get_variable(
                'w',[5,5,64,128],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias2=tf.get_variable(
                'b',shape=[128],
                initializer=tf.zeros_initializer())
            conv2=tf.nn.conv2d(out1,weight2,[1,2,2,1],'SAME')
            out_add2=tf.add(conv2,bias2)
            mean2,var2=tf.nn.moments(out_add2,[0,1,2])
            out_norm2=tf.nn.batch_normalization(out_add2,mean2,var2,None,None,1e-5)
            out2=tf.nn.leaky_relu(out_norm2)

        with tf.variable_scope('conv3',reuse=reuse):
            weight3=tf.get_variable(
                'w',[5,5,128,256],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias3=tf.get_variable(
                'b',shape=[256],
                initializer=tf.zeros_initializer())
            conv3=tf.nn.conv2d(out2,weight3,[1,2,2,1],'SAME')
            out_add3=tf.add(conv3,bias3)
            mean3,var3=tf.nn.moments(out_add3,[0,1,2])
            out_norm3=tf.nn.batch_normalization(out_add3,mean3,var3,None,None,1e-5)
            out3=tf.nn.leaky_relu(out_norm3)

        with tf.variable_scope('conv4',reuse=reuse):
            weight4=tf.get_variable(
                'w',[5,5,256,512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias4=tf.get_variable(
                'b',shape=[512],
                initializer=tf.zeros_initializer())
            conv4=tf.nn.conv2d(out3,weight4,[1,2,2,1],'SAME')
            out_add4=tf.add(conv4,bias4)
            mean4,var4=tf.nn.moments(out_add4,[0,1,2])
            out_norm4=tf.nn.batch_normalization(out_add4,mean4,var4,None,None,1e-5)
            out4=tf.nn.leaky_relu(out_norm4)

        reshape=tf.reshape(out4,[out4.get_shape()[0].value,-1]) #batchを除く次元でflatten

        with tf.variable_scope('fully_connect',reuse=reuse):
            weight5=tf.get_variable(
                'w',[4*4*512,2],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias5=tf.get_variable(
                'b',[2],
                initializer=tf.zeros_initializer())
            out5=tf.add(tf.matmul(reshape,weight5),bias5)


    return out5

batch_size=64
inputs=tf.random_normal([batch_size,100])
real=batch(batch_size)
fake=generator(inputs,batch_size)
real_logits=disctiminator(real)
fake_logits=disctiminator(fake,reuse=True)

g_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.ones([batch_size],dtype=tf.int64),
    logits=fake_logits))
d_loss=tf.reduce_sum([
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.zeros([batch_size],dtype=tf.int64),
        logits=fake_logits)),
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.ones([batch_size],dtype=tf.int64),
        logits=real_logits))])

g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='g')
d_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='d')

g_train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss,var_list=g_vars)
d_train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss,var_list=d_vars)

generated=tf.concat(tf.split(fake,batch_size)[:8],2)
generated=tf.divide(tf.add(tf.squeeze(generated,axis=0),1.0),2.0)
generated=tf.image.convert_image_dtype(generated,tf.uint8)
output_img=tf.image.encode_png(generated)

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)

    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        _, _, g_loss_value, d_loss_value=sess.run([g_train_op,d_train_op,g_loss,d_loss])
        print('step{:5d}:g={:.4f},d={:4f}'.format(i+1,g_loss_value,d_loss_value))

        if i%100==0:
            img=sess.run(output_img)
            with open(os.path.join(os.path.dirname(__file__),'{:05d}.png'.format(i)),'wb') as f:
                f.write(img)
    
    coord.request_stop()
    coord.join(threads)




        




            
