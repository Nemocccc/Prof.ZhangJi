import custom_envs
import gymnasium as gym

if __name__ == "__main__":
    env=gym.make("custom_envs/TVM-v0")
    done=False
    obs=env.reset(seed=35)

    while not done:
        action=env.action_space.sample()
        obs,reward,terminated,truncated,info=env.step(action)
        done=terminated or truncated
        env.render()
        
