from gymnasium.envs.registration import register
 
register(
     id="custom_envs/TVM-v0",
     entry_point="custom_envs.tvm_game:TVM",
     max_episode_steps=200,
)

register(
     id="custom_envs/TVM_re-v0",
     entry_point="custom_envs.re_tvm_game:TVM",
     max_episode_steps=200,
)

register(
     id="custom_envs/RVB-v0",
     entry_point="custom_envs.rvb_game:TVM",
     max_episode_steps=200,
)