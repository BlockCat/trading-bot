from gym.envs.registration import register

print('registered stockenv-v0')
register(
    id='stockenv-v0',
    entry_point='stock_environment.gym:StockTradingEnvironment',
)
