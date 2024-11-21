from gym.envs.registration import register

register(
    id='Lattice3D-6actionStateEnv-v0',
    entry_point='gym_lattice.envs:FiveActionStateEnv',
)
