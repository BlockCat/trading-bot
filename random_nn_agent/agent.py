import gym
import tensorflow as tf

from stock_environment.BaseAgent import BaseAgent,
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class Agent(BaseAgent):

    def get_weight_path(self, name: str):
        return f'temp/weights-{name}.h5f'

    def create_model(self, input_size: int):
        # input_placeholder = tf.keras.backend.placeholder(shape=(None, None, input_size))
        # output_placeholder = tf.keras.backend.placeholder(shape=(None, output_size))

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.env.observation_space.shape))     
        model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.MaxPool1D(2))
        model.add(tf.keras.layers.LSTM(4))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100))
        model.add(tf.keras.layers.Dense(self.env.action_space.n))
        model.compile(
            loss='mse'
        )        
        print(model.summary())
        return model
