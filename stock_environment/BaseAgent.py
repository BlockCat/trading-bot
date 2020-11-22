import gym
import wandb

from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback, ModelIntervalCheckpoint, FileLogger
from stock_environment.wandb_callback import WandbLogger
from stock_environment.stock_processor import StockProcessor
from data_reader import read_daily_data

WINDOW_LENGTH = 20


class BaseAgent:

    dqn: DQNAgent

    def __init__(self, stock: str):
        self.env = gym.make('stockenv-v0', df = read_daily_data(stock))
        self.env.seed(123)
        self.stock = stock

        memory = SequentialMemory(
            limit=1000000,
            window_length=WINDOW_LENGTH
        )
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr='eps',
            value_max=1.,
            value_min=.1,
            value_test=.05,
            nb_steps=1000000
        )

        processor = StockProcessor(stock)

        self.dqn = DQNAgent(
            model=self.create_model(30),
            nb_actions=self.env.action_space.n,
            policy=policy,
            memory=memory,
            processor=processor,
            nb_steps_warmup=50000,
            gamma=.99,
            target_model_update=10000,
            train_interval=4,
            delta_clip=1.
        )
        self.dqn.compile(Adam(lr=.00025), metrics=['mae'])

    def train(self):

        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = self.get_weight_path(self.stock)
        checkpoint_weights_filename = self.get_weight_path(
            self.stock) + '_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(self.stock)

        callbacks = [ModelIntervalCheckpoint(
            checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        callbacks += [WandbLogger(
            project = "stock-bot-v0"
        )]

        self.dqn.fit(self.env, callbacks=callbacks,
                     nb_steps=1750000, log_interval=10000)

        # After training is done, we save the final weights one more time.
        self.dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        self.dqn.test(self.env, nb_episodes=10, visualize=False)

    def test(self):
        weights_filename = self.get_weight_path(self.stock)
        self.dqn.load_weights(weights_filename)
        self.dqn.test(self.env, nb_episodes=10, visualize=True)

    def get_weight_path(self, name: str) -> str:
        """Get weight path"""
        pass

    def create_model(self, input_size: int):
        """abstract"""
        pass
