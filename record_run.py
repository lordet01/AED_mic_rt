import pyaudio
import wave
import librosa
import soundfile as sf
import numpy as np
import tensorflow as tf
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
FRAME_LENGTH = 512
N_MFCC = 16
N_BLOCK = 16
#RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "record.pcm"

audio = pyaudio.PyAudio()


### Start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=FRAME_LENGTH)
print ("recording...")


### Save recorded file into pcm file
audio_file = open(WAVE_OUTPUT_FILENAME, "ab") 


### Load NNet Model
drate = 1.0
lrate = 0.0001
ksize = [5,5]
ssize = [2,2]
nfilter = 3
batch_size = 50
total_batch = 100

x_ = tf.placeholder(tf.float32, [None, N_MFCC, N_BLOCK, 1])
y_ = tf.placeholder(tf.float32, [None, 3])

conv1_ = tf.layers.conv2d(inputs=x_, filters=nfilter, kernel_size=ksize, strides=ssize, padding='SAME', activation=tf.nn.relu)
drop1_ = tf.layers.dropout(inputs=conv1_, rate=drate)

conv2_ = tf.layers.conv2d(inputs=drop1_, filters=2*nfilter, kernel_size=ksize, strides=ssize, padding='SAME', activation=tf.nn.relu)
drop2_ = tf.layers.dropout(inputs=conv2_, rate=drate)

conv3_ = tf.layers.conv2d(inputs=drop2_, filters=3*nfilter, kernel_size=ksize, strides=ssize, padding='SAME', activation=tf.nn.relu)
drop3_ = tf.layers.dropout(inputs=conv3_, rate=drate)

flat_ = tf.reshape(drop3_, [-1, 2 * 2 * 9])
dense1_ = tf.layers.dense(inputs=flat_, units=36, activation=tf.nn.relu)

hypothesis = tf.layers.dense(inputs=dense1_, units=3, activation=tf.nn.softmax)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y_))
train = tf.train.AdamOptimizer(learning_rate=lrate).minimize(cost)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './my-model-20180402-1000')

decision = tf.argmax(hypothesis, 1)


### Debug file
f_debug = open('feat_in.txt', 'a')

x = np.zeros((FRAME_LENGTH, 1), dtype='int16')
x_seq = np.zeros((FRAME_LENGTH*N_BLOCK, 1), dtype='float32')
feat_in = np.zeros((N_MFCC, N_BLOCK), dtype='float32')
seq_cnt = 0
while True:
	audio_frame = stream.read(FRAME_LENGTH) #Read Audio Frame
	audio_file.write(audio_frame) #Write Audio Frame to pcm file
	
	### Extract each channel
	x = np.frombuffer(audio_frame, dtype=np.int16) #Convert Byte to short(int16)
	x = np.stack((x[::2], x[1::2]), axis=1)
	x = np.mean(x, axis=1) # Down-mix to mono from stereo (averaging)

	### Stack frames into buffer
	x_seq = np.concatenate((x_seq[FRAME_LENGTH:], x), axis=None)
	seq_cnt = seq_cnt + 1

	## Block-wise Process Begins
	if seq_cnt == N_BLOCK:
		# Calculate MFCC features from the raw signal
		mfcc = librosa.feature.mfcc(y=x_seq, sr=RATE, hop_length=FRAME_LENGTH, n_mfcc=N_MFCC)
		mfcc = mfcc[:, 0:N_BLOCK]
		np.savetxt(f_debug, mfcc)
		feat_in = np.reshape(mfcc, (1, N_MFCC, N_BLOCK,1))
		#print(np.shape(feat_in))

		#Run TF Inference
		pred_ret = sess.run(hypothesis, feed_dict={x_: feat_in})
		dec_ret = sess.run(decision, feed_dict={x_: feat_in})

		print(dec_ret)
		seq_cnt = 0






	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	