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
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=FRAME_LENGTH)
print ("recording...")
 
#save recorded file into pcm file
audio_file = open(WAVE_OUTPUT_FILENAME, "ab") 

#Load NNet Model
##saver.restore(sess, './my-model-20180402-1000')
##decision = tf.argmax(hypothesis, 1)

#Files for debugging
f = open("debug.txt", 'a')
 
#for i in range(0, int(RATE / FRAME_LENGTH * RECORD_SECONDS)):
x = np.zeros((N_MFCC,1), dtype='int16')
x_blk = np.zeros((N_MFCC, N_BLOCK), dtype='float32')
while True:
	audio_frame = stream.read(FRAME_LENGTH) #Read Audio Frame
	audio_file.write(audio_frame) #Write Audio Frame to pcm file
	
	x = np.frombuffer(audio_frame, dtype=np.int16) #Convert Byte to short(int16)
	x = [x[::2], x[1::2]] #Separate to each channel
	np.transpose(x)
	np.mean(x, axis=1) # Down-mix to mono from stereo (averaging)
	
	# Calculate MFCC features from the raw signal
	audio_frame = audio_frame.astype('float32') #Convert data type from int to float
	mfcc = librosa.feature.mfcc(y=audio_frame, sr=RATE, hop_length=FRAME_LENGTH, n_mfcc=N_MFCC)

	# Stack frames into buffer
	x_blk = [x_blk[:, 1:], mfcc]
	f.write(x_blk)

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	