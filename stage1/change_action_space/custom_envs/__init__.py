from gymnasium.envs.registration import register

register(
     id="TVM-v3",
     entry_point="custom_envs.tvm_v3.tvm_v3:TVM",
     max_episode_steps=200,
)

register(
     id="TVM-v4",
     entry_point="custom_envs.tvm_v4.tvm_v4:TVM",
     max_episode_steps=200,
)


register(
     id="TVM-v5",
     entry_point="custom_envs.tvm_v5.tvm_v5:TVM",
     max_episode_steps=200,
)

register(
     id="TVM-v1",
     entry_point="custom_envs.tvm_v1.tvm_v1:TVM",
     max_episode_steps=200,
)

register(
     id="TVM-v2",
     entry_point="custom_envs.tvm_v2.tvm_v2:TVM",
     max_episode_steps=200,
)

register(
     id="Maze-v0",
     entry_point="custom_envs.maze.maze:MazeEnv",
     max_episode_steps=300,
)


register(
     id="Combat-v0",
     entry_point="custom_envs.combat.combat:VisualMABattleEnv",
     max_episode_steps=-1,
)

