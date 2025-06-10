from gymnasium.envs.registration import register

register(
    id = 'Blue_env-v0',
    entry_point = 'envs.Blue_env_v0:BlueEnv'
)
