import minari

dataset = minari.load_dataset("pendulum/sac-v0")
dataset.set_seed(seed=123)

for episode_data in dataset.iterate_episodes():
    observations = episode_data.observations
    actions = episode_data.actions
    rewards = episode_data.rewards
    terminations = episode_data.terminations
    truncations = episode_data.truncations
    infos = episode_data.infos
    print(f"Episode ID: {episode_data.id}")
    print(f"Observations: {len(observations)}")
    print(f"Actions: {len(actions)}")
    # print(f"Rewards: {rewards}")