import gym
import tensorflow as tf

from stock_environment.BaseAgent import BaseAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class Agent(BaseAgent):

    def get_weight_path(self, name: str):
        return f'temp/weights-{name}.h5f'

    def create_model(self, input_size: int):
        # input_placeholder = tf.keras.backend.placeholder(shape=(None, None, input_size))
        # output_placeholder = tf.keras.backend.placeholder(shape=(None, output_size))

        self.env.action_space
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.Flatten())        
        model.add(tf.keras.layers.Dense(4))
        model.compile(
            loss='mse'
        )
        model.build(input_shape=(None, None, input_size))
        print(model.summary())
        return model
