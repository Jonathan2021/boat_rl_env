install with

`python setup.py build`

`pip install -e .`

a self explanatory-ish implementation example is given in human_agent.py 


# The various environments

We have various envrionment, compliant to the OpenAI Gym API and registered to OpenAI Gym in shipNavEnv.\_\_init\_\_
They are accessible with the name ShipNav-vX.

* **v0**: Only rocks as obstacle. Actions are steer only and are discrete. Ship agent doesn't have lidars but can use radar image (and an old radar version if specified in the parameters)
* **v1**: Same condiations as v0 but the agent has lidars (doesn't use radar). Steering only and is discrete.
* **v2**: Obstacles are ships only and no lidar on the agent (uses radar). Steering only and is discrete. <- Never really trained on
* **v3**: Same as v0 but steering is continuous. <- Never trained on
* **v4**: Same as v3 but actions also allow to throttle in a continuous fashion <- Never trained on
* **v5**: Same as v1 but obstacles are ships only.
* **v6**: Same as v5 but can also use the radar.
* **v7**: The last and most complete env yet. Action is steering and is discrete. Obstacles are ships and rocks. Ship can use both lidar and radar.

Note that each time radar is available, the old version of the radar (not an image but a list of positions, speeds etc.) is also available. You just need to set the n_obstacles_obs argument to be something greater than 0.

# The state
The state is composed of several parts and uses OpenAI soaces.Dict to seperate cleanly these parts. All states are standardized so that they take values in [-1, 1] with a mean of around 0 (except for the radar image which are pixels in 0, 255).
Everything is in the frame of the agent (And not the world frame).

* **Ship (agent) state**:
    * Agent's x axis speed
    * Agent's y axis speed
    * Agent's angular velocity
    * Agent's thruster angle
    * Agent's distance to objective (target or waypoint)
    * Agent's bearing to objective (target or waypoint)
    * Lidar fractions (if ship has lidars)
* **World state**
Fraction of the remaining time before time limit is reached and episode is stopped.
* **Obstacles state (old radar)**:
This is the old radar (tabular) and is off by default. It onyl detects a predefined number of obstacles.
Rocks give distance and bearing from the agent.
Ships give distance, bearing, x speed, y speed, angular velocity and bearing from ship. This is pretty deprecated but I kept it in case we manage to make it work as well as the new radar because it is a lot faster.
* **Radar image state**
Image of the obstacles in a certain area. The image is in the agent's frame (first person). Obstacles are drawn, the last 5 (editable) positions of the ships are recorded as dots (recorded every second but also changeable).
Color channels are used to encode information (red is x speed, green is y speed and blue is angular velocity). Dots keep the color they had when created.

# **Reward function**
The reward function should guide the agent without introducing a bias. There is a tradeoff between sparsity (reward only when succeeding and failing) and bias (we want to help it go in the "correct" direction but by doing lead it to a suboptimal behaviour). The reward should vary between (approximatly -5/-6 and +1)

The last reward to date is, at every timestep:


* `-1 / max number of timesteps` <- Can be viewed as fuel consumption penalty and forces the agent to not waste too much time. Removing it may lead to agent turning in circle because going to target is too risky and it would rather not get a reward at all.

* `-abs(self.world.ship.thruster_angle / self.world.ship.THRUSTER_MAX_ANGLE) / self.MAX_STEPS`. This is divided by 2 if the angle is to the right (making it turn to starboard). This division by 2 is the only hack I could think of to make it favor going starboard (which is often prefered in COLREGs)

* `sqrt(n_touches) * -1 / self.MAX_STEPS` where n_touches are the number of bodies (their sensor or physical body) touched by the bumper around the ship (capsule shape). It is kinda hacky but if touches something then >= timestep reward and if touches more then punishes more without penalizing too much envs satured in obstacles where it is impossible to not touch anything.

* -5 if you hit something

* -1 if you get to the end of the episode

* +1 if you reach the target

I used to have a reward proportional to the delta distance to target (dist t-1  - dist t)

# **The obstacles**
* Rocks:
They are circular static obstacles of varying radius.
* Ships:
Same as the agent's ship in shape. 95% of the ships follow a handwritten behavior (which tries to be colreg compliant but this still needs some work done) and 5% have a random behavior. They can slow down as well as steer.
The wanted behavior of the handwritten AI is the following:
    * Follow a certain angle (world frame).
    * If something in front slow down and turn right
    * If something coming / on starboard side. Turn right.
    * If nothing in the way and deviating from the angle it wants to follow: rectify trajectory to follow this angle.


# Questions you may ask yourself
* ## Why do we have a scale sometimes ?
The scale is used to allow obstacles to be placed outside the original map size so that the agent doesn't just learn to go around the obstacles. This is to make it appear as if the map with obstacles is infinite.
* ## What are the waypoints ?
You can activate or not waypoint support with the argument `waypoints = True/False`. These waypoints mark a safe path for the ship to follow (only takes terrain = rocks into account + scales them so that the path doesn't go too close) and the current waypoint replaces the target until it is reached and so on. This was implemented before generating the radar image where the agent sometimes chose difficult paths because it didn't have a good understanding of the spatial arrangement of the terrain. Typically you would train without waypoints and then evaluate with and without to estimate how much choosing a safe route matters for the agent. I haven't used it in a while.
* ## Why is it slow ?
The physics simulation (and most importantly collisions) and the image generation (radar) slow things down a lot.
* ## What is the virtual display
We use a hidden virtual display to render the state so that it doesn't appear on the screen. We don't want it to show during training but we still need a display to draw on to generate the image with gym rendering.
* ## Why leave a trail behind ships
Helps the agent decipher the trajectories of obstacles while keeping the speeds and angular velocities at the time they were recorded, giving also information about the ships own trajectory). This gives info about the past and makes the state more compliant with the Markov property.
* ## What is the use of the bumpers ?
The idea came from the paper [Automatic ship collision avoidance using deep reinforcement learning with LSTM in continuous action spaces](https://link.springer.com/article/10.1007/s00773-020-00755-0). It is used for reward in the case of our agent and for writting the AI for ship obstacles. It can be considered as a safe zone around the ships.


# THE TODO CONTAINS THINGS TO DO AS WELL AS IDEAS, TIPS, BIBLIO ETC.