from rl.core import Processor


class StockProcessor(Processor):
    def __init__(self, stock: str):
        self.stock = stock

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.

            # Arguments
                action (int): Action given to the environment

            # Returns
                Processed action given to the environment
            """
        print("Action chosen:" + str(action))
        return action
