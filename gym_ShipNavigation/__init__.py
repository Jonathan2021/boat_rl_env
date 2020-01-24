from gym.envs.registration import register

register(
    id='ShipNavigation-v0',
    entry_point='gym_ShipNavigation.envs:ShipNavigationEnv',
)
register(
    id='ShipNavigationLidar-v0',
    entry_point='gym_ShipNavigation.envs:ShipNavigationLidarEnv',
)
register(
    id='ShipNav-v0',
    entry_point='gym_ShipNavigation.envs:ShipNavigationWithModelEnv',
)