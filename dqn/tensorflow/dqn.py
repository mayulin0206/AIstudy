import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np

REPLAY_SIZE = 10000
SMALL_BATCH_SIZE = 16
BIG_BATCH_SIZE = 128
BATCH_SIZE_DOOR = 1000

GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01

class DQN():
	def __init__(self,observation_space,action_sapce):
		self.state_dim = observation_space.shape[0]
		self.action_dim = action_sapce.n
		self.replay_buffer = deque()
		self.create_Q_network()
		self.create_updating_method()
		self.epsilon = INITIAL_EPSILON
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())


	def create_Q_network(self):
		self.state_input = tf.placeholder("float",[None,self.state_dim])
		with tf.variable_scope('current_net'):
			w1 = self.weight_variable([self.state_dim,50])
			b1 = self.bias_variable(50)
			w2 = self.weight_variable([50,20])
			b2 = self.bias_variable([20])
			w3 = self.weight_variable([20,self.action_dim])
			b3 = self.bias_variable([action_dim])

			h_layer_one = tf.nn.relu(tf.matmul(self.state_input,w1) + b1)
			h_layer_two - tf.nn.relu(tf.matmul(h_layer_one,w2) + b2)

			self.q_value = tf.matmul(h_layer_two,w3) + b3

		with tf.variable_scope('target_net'):
			t_w1 = self.weight_variable([self.state_dim,50])
			t_b1 = self.bias_variable(50)
			t_w2 = self.weight_variable([50,20])
			t_b2 = self.bias_variable([20])
			t_w3 = self.weight_variable([20,self.action_dim])
			t_b3 = self.bias_variable([action_dim])

			t_h_layer_one = tf.nn.relu(tf.matmul(self.state_input,t_w1) + t_b1)
			t_h_layer_two - tf.nn.relu(tf.matmul(t_h_layer_one,t_w2) + t_b2)

			self.t_q_value = tf.matmul(t_h_layer_two,t_w3) + t_b3

		e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIBALES,scope = 'current_net')
		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIBALES,scope = 'target_net')

		with tf.variable_scope('soft_replacement'):
			self.target_replace_op = [tf.assign(t,e) for t, e in zip(t_params,e_params)]


	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape=shape)
		return tf.Variable(initial)

	def create_updating_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim])
		self.y_input = tf.placeholder("float",[None])
		Q_action = tf.reduce_sum(tf.multiply(self.q_value,self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def choose_action(self,state):
		q_value = self.q_value.eval(feed_dict = {self.state_input: [state]})[0]
		if random.random() <= self.epsilon:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
			return random.randint(0,self.action_dim - 1)
		else:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
			return np.argmax(q_value)

	def store_data(self,state,action,reward,next_state,done):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state,action,reward,next_state,done))
		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()

	def train_network(self,BATCH_SIZE):
		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		y_batch = []
		q_value_batch = self.t_q_value.eval(feed_dict = {self.state_input:next_state_batch})

		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

		self.optimizer.run(feed_dict = {
			self.y_input: y_batch,
			self.action_input: action_batch,
			self.state_input: state_batch
			})

	def update_targer_network():
		self.session.run(self.target_replace_op)

	def action(self,state):
		return np.argmax(self.q_value.eval(feed_dict= {self.state_input:state})[0])

ENV_NAME = 'CartPole-v0'
EPISODES = 1000
STEPS = 300
UPDATE_STEP = 50
TEST = 5

def main():
	env = gym.make(ENV_NAME)
	agent = DQN(env.observation_space,env.action_space)
	for episode in range(EPISODES):
		state = env.reset()
		for step in range(STEPS):
			action = agent.choose_action(state)
			next_state,reward,done,info = env.step(action)
			agent.store_data(state,action,reward,next_state,done)
			if len(agent.replay_buffer) > BIG_BATCH_SIZE:
				agent.train_network(BIG_BATCH_SIZE)
			if step % UPDATE_STEP == 0:
				agent.update_targer_network()
			state = next_state
			if done:
				break

		if episode % 100 == 0:
			total_reward = 0;
			for i in range(TEST):
				state = env.reset()
				for j in range(STEPS):
					env.render()
					action = agent.action(state)
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward / TEST
			print('episode:', episode, 'Evaluation Average Reward:', ave_reward)

if __name__ == '__main__':
	main()

