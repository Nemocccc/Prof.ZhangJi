import custom_envs
import gymnasium as gym
import time
from test_agent_0 import *
if __name__ == "__main__":
    env=gym.make("Maze-v0",render_mode="rgb_array")
    done=False
    
    options={'monster_num':6}
    raw_obs,info=env.reset(seed=32,options=options)

    # obs=env.reset(seed=35)

    while not done:
        action=env.action_space.sample()
        raw_obs,reward,terminated,truncated,info=env.step(action)
        
        # exit(0)
        print(reward)
        
        # obs_pipeline(env,raw_obs,encode_mode=1)

        done=terminated or truncated
        image=env.render()
        print(image)
        
        print(action)

