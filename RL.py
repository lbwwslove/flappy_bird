from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np
import random

"""
Here are two values you can use to tune your Qnet
You may choose not to use them, but the training time
would be significantly longer.
Other than the inputs of each function, this is the only information
about the nature of the game itself that you can use.
"""
PIPEGAPSIZE  = 100
BIRDHEIGHT = 24

class QNet(object):

	def __init__(self):
		"""
		Initialize neural net here.
		You may change the values.

		Args:
			num_inputs: Number of nodes in input layer
			num_hidden1: Number of nodes in the first hidden layer
			num_hidden2: Number of nodes in the second hidden layer
			num_output: Number of nodes in the output layer
			lr: learning rate
		"""
		self.num_inputs = 1
		self.num_hidden1 = 10
		self.num_hidden2 = 10
		self.num_output = 2
		self.lr = 0.05
		self.build()
		self.k = 300
		self.count = 0
		self.old_states = []
		self.updated_pred = []
		self.score_count = 0
		# This cap is to stop fitting the model if our bird can pass 4 pipes, if the bird can't pass
		# more than 4 pipes, we keep fitting. This number of 4 is observed. My bird will consistently 
		# pass 100+ pipes if it's able to pass the first 4
		self.score_count_cap = 4

	def build(self):
		"""
		Builds the neural network using keras, and stores the model in self.model.
		Uses shape parameters from init and the learning rate self.lr.
		You may change this, though what is given should be a good start.
		"""
		model = Sequential()
		model.add(Dense(self.num_hidden1, init='lecun_uniform', input_shape=(self.num_inputs,)))
		model.add(Activation('relu'))

		model.add(Dense(self.num_hidden2, init='lecun_uniform'))
		model.add(Activation('relu'))

		model.add(Dense(self.num_output, init='lecun_uniform'))
		model.add(Activation('linear'))

		rms = RMSprop(lr=self.lr)
		model.compile(loss='mse', optimizer=rms)
		self.model = model


	def flap(self, input_data):
		"""
		Use the neural net as a Q function to act.
		Use self.model.predict to do the prediction.

		Args:
			input_data (Input object): contains information you may use about the 
			current state.

		Returns:
			(choice, prediction, debug_str): 
				choice (int) is 1 if bird flaps, 0 otherwise. Will be passed
					into the update function below.
				prediction (array-like) is the raw output of your neural network,
					returned by self.model.predict. Will be passed into the update function below.
				debug_str (str) will be printed on the bottom of the game
		"""
		state = np.array([input_data.distY])
		prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size=1)[0]

		if (prediction[0] >= prediction[1]):
			choice = 1
		else:
			choice = 0

		debug_str = str(prediction)
		return (choice, prediction, debug_str)


	def update(self, last_input, last_choice, last_prediction, crash, scored, playerY, pipVelX):
		"""
		Use Q-learning to update the neural net here
		Use self.model.fit to back propagate

		Args:
			last_input (Input object): contains information you may use about the
				input used by the most recent flap() 
			last_choice: the choice made by the most recent flap()
			last_prediction: the prediction made by the most recent flap()
			crash: boolean value whether the bird crashed
			scored: boolean value whether the bird scored
			playerY: y position of the bird, used for calculating new state
			pipVelX: velocity of pipe, used for calculating new state

		Returns:
			None
		"""

		new_distX = last_input.distX + pipVelX
		new_distY = last_input.pipeY - playerY
		state = np.array([new_distY])
		gamma = 0.1
		# reward = compute your reward
		if (crash == 1):
			reward = -abs(new_distY)
		else:
			if (new_distY >= BIRDHEIGHT and new_distY <= PIPEGAPSIZE):
				reward = new_distY - BIRDHEIGHT
				

			if (new_distY < BIRDHEIGHT and last_choice == 1):
				reward = abs(BIRDHEIGHT - new_distY)
			if (new_distY < BIRDHEIGHT and last_choice == 0):
				reward = -abs(BIRDHEIGHT - new_distY)
			if (new_distY > PIPEGAPSIZE and last_choice == 0):
				reward = new_distY - PIPEGAPSIZE
			if (new_distY > PIPEGAPSIZE and last_choice == 1):
				reward = -(new_distY - PIPEGAPSIZE)
		
		# if (last_input.distY <= 40):
		# 	if (last_choice == 1):
		# 	    reward = 20
		# 	else:
		# 		reward = -10
		# else:
		# 	if (last_choice == 1):
		# 		reward = -10
		# 	else:
		# 		reward = 20



		prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size = 1)

		# update old prediction from flap() with reward + gamma * np.max(prediction)
		last_prediction[not last_choice] = reward + gamma * np.max(prediction)
		self.old_states.append(np.array([last_input.distY]))
		self.updated_pred.append(last_prediction)
		self.count += 1
		if (scored):
		    self.score_count += 1
		if (crash):
			self.score_count = 0

		if (self.count == self.k):
			temp1 = random.sample(self.old_states, self.k)
			temp1 = np.asarray(temp1)
			temp2 = random.sample(self.updated_pred, self.k)
			temp2 = np.asarray(temp2)
			if (self.score_count < self.score_count_cap):
			    self.model.fit(temp1, temp2, batch_size=self.k, epochs=1)
			self.count = 0
			self.old_states = []
			self.updated_pred = []


        # record updated prediction and old state in your mini-batch
        # if batch size is large enough, back propagate
        # self.model.fit(old states, updated predictions, batch_size=size, epochs=1)
		
class Input:
	def __init__(self, playerX, playerY, pipeX, pipeY,
				distX, distY):
		"""
		playerX: x position of the bird
		playerY: y position of the bird
		pipeX: x position of the next pipe
		pipeY: y position of the next pipe
		distX: x distance between the bird and the next pipe
		distY: y distance between the bird and the next pipe
		"""
		self.playerX = playerX
		self.playerY = playerY
		self.pipeX = pipeX
		self.pipeY = pipeY
		self.distX = distX
		self.distY = distY

