# TODO / IDEAS
* Read the code and fix all FIXMEs and do all TODOs
* Make a better handwritten AI for ship obstacles to obey COLREGs.
* Find a good way of introducing COLREGs in the reward. (at least a better way than what is used now)
* Step a few times in the simulation in init so that the first state doesn't have 0 speed etc.
* Add stochasticity in initial speeds, angular velocities, thruster angles etc.
* Big obstacle (island like) -> [How to calculate distance](https://box2d.org/files/ErinCatto_GJK_GDC2010.pdf)
* Try model-based RL (ship predicts next position or other)
* Try setting damping directly in box2D (just a factor and then it does calculation itself)
* Center of mass calculated by engine from fixtures instead of by hand
* Make a ship body a bit more realistic (several fixtures etc. to get correct center of mass...) with B2Draw if needed
* Make rock obstacles asleep and others sleepable (when possible) to gain computational speed
* Add CPA (closest point of approach) in state somehow maybe ? <- Agent probably finds something like this by itself
* Add OZT support maybe (as in this paper [Automatic ship collision avoidance using deep reinforcement learning with LSTM in continuous action spaces](https://link.springer.com/article/10.1007/s00773-020-00755-0)) ? <- Agent probably comes up with something similar automatically
* Make values more realistic for ships
* Make the simulation 3D and more photo relaistic
* Make other ships different in size, speeds etc. (and find a way to normalize values somehow for state)
* Add frame stacking to hyperparam optimization.
* Allow skipping frames when frame stacking (because stacking 10 frames in a row gives 1/3 second of info which is maybe not really useful
* Try using recurrency in networks (extend stable baselines class with custom network since not supported yet)
* Try multiagent reinforcement learning
* Try pretraining with behavior cloning (imitation learning)
* Add type definitions in function signatures for speedup
* Add ships that don't move at all in the middle of the see
* Only add hit_with etc. to ship we are interested in, not others or rocks etc. to gain memory
* Try to implement hyperparams optimization with pretrained agent (because once trained for a while, some other parameters may be optimal + you might want to change the env distribution etc. in the middle of training ...)
* Train with continuous action spaces and with steering and throttle control.
* Make a benchmark env or framework (Have some random envs + some predefined configurations that have to work (see Imazu problems as they have done in [Automatic ship collision avoidance using deep reinforcement learning with LSTM in continuous action spaces](https://link.springer.com/article/10.1007/s00773-020-00755-0)))
* Check if stacking frames make a difference in performance (if yes we are probably missing info from the past in the current state -> not markovian)
* Don't update all obstacle data, just the ones needed
* Possibility of having separate radius for old and new radar
* Clean up code (perhaps only one big base class where everything is just parameters and the envs are a specific combo of parameters...)
* Fix weird lidar angle at first step (not sure if it is still the case but when rendering, the first image was weird)
* Seen logic should be handled in env, not in world with body.can_see ?
* Fix n_jobs core dump when doing multithread hyperparameter optimization
* Dig into env wrappers and use them when useful
* Dig into callbacks and use them when useful
* Look into Procgen to make simulation faster
* I think boat still spawns inside target sometimes but very rare
* Add stochasticity/noise to states to make agent more robust
    * keep same action k-steps
    * blink screen
    * Make some obstacles disappear
    * fuzzy sensors
    * Make some ships in radar not have their speed etc. (grey as if not moving when they actually are)
    * perturbate action

* Render 2 things when we have noise
    * Video of what is actually hapenning
    * Video of what the boat thinks is happening
* Look into using GAIL and AIRL to build a reward function
* Close Xvfb daemons somehow (opened by virtual display and are still there after "closing" them which in facts just closes their port)
* Don't forget to normalize states if you add some !
* Wrapper to normalize reward ? [why you should perhaps ?](https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem)
* Remove repetitions in init and reset, reset and destroy etc.
* Maybe add the agent's trajectory to state somehow (in radar image or some other way)
* Check that image is not too small (for radar) -> bigger made hyperparam research run out of memory
* Remove bumper from state image since it leaks info about ship behavior -> it helps in knowing ship orientation (which is not obvious from its shape, especially in small image) so maybe replace ship shape by a triangle or something more visible (like an arrow in front) -> or keep the bumper with dimension as "average" but with the real sensors being invisible and with an std around the mean so that it doesn't learn exactly the ship obstacle behavior**
* To test importance of radar, replace ship with random image etc.
* Try removing lidars
* Maybe use some regularization and analyze network weights
* Radar image directly as a matrix to make training faster
* Test more algorithms
* Don't forget to fetch and merge the main repo for Stable Baselines3 Zoo regularly !
* Consider using [Pybullet](https://github.com/bulletphysics/bullet3/) for 3D physics simulation ? Or something eles
* Dig into and try using [evolution strategies](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/)
* In practice, you have already have a rough map of the terrain (coast line etc.). This could probably be fed to the AI. -> Suppose it knows the target coordinates, but there is a big piece of land in between : should it go left or right ? (Depends on how far it stretches on each sides). -> Could be done with radar like technique but only for fixed obstacles.
* Add water currents, waves ... ?
* Larger punishment if hard impact ?
* Maybe something like hyperparameter opt but to see which env params yield the best results on the benchmark test ?
* Several agents that interact, for example one for choosing optimal path, one to follow this path closely. Or one to control where to scan etc.
* Model that automatically selects ships / obstacles to observe instead of using image
* Better reward doesn't necessarily mean better agent. Make sure they mean the same. (maybe choose hyperparams based on best success rate instead etc.)
* Add checkpoints while training (supported by SB3 zoo framework)

# TIPS

* ## PREPROCESSING

 * Normalize the input to the agent (e.g. using VecNormalize for PPO/A2C)
 * Look at common preprocessing done on other environments (e.g. for Atari, frame-stack, …)

* ## How to evaluate an RL algorithm
* [SB3 doc](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm)
* [SB3 'Advanced Saving and Loading'](https://github.com/bulletphysics/bullet3/)

* ## Which algorithm should I use ?
* [SB3 doc](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#which-algorithm-should-i-use)
* [SB3 doc on implemented RL algorithms](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)

* ## Creating a custom environment
* [SB3 doc](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment)
* [OpenAI doc](https://github.com/openai/gym/blob/master/docs/creating-environments.md)

* ## When implementing an RL algorithm
* [SB3 doc](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-implementing-an-rl-algorithm)

* ## EXAMPLES
* [A bunch of example available SB3 doc](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)

* ## CALLBACK
* [SB3 doc](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training)
* [more SB3 doc](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)

# Similar works

[Hierarchical multiagent reinforcement learning for maritime traffic management management (2020)](https://core.ac.uk/download/pdf/361929058.pdf) **High level**:

- Multi-agent + Regulatory agent
- Decentralized control/policies which models the uncertainty and partial observability (vessels lern decentralized policy from local observations)
- use of meta actions and optimizing a policy over them
- hierarchical reinforcement learning
- State:  If vesselmis in zonez,andntott=⟨ntott(z)∀z⟩be the count table representing total numberof vessels present in different zones (we show how to computeit later), then agentm’s observation iso(z,ntott). Typically, in apartially observable setting, this observation corresponds to thecounts of all vessels in zonezand local neighborhood ofz
- Action: When a vesselmis newly-arrived at azonez, it needs to take two actions—direction actionamtto decidewhich zonez′to go to next; andnavigation actionωmtto decidehow much time to take to navigate toz′. When a vessel is in-transit (orsmt=⟨z,z′,τ⟩), it can only take adummydirection action.

[A SURVEY OF MACHINE LEARNING APPROACHES FOR SURFACE MARITIME NAVIGATION](https://upcommons.upc.edu/bitstream/handle/2117/329714/08_Porres.pdf?sequence=1&isAllowed=y) **Overview**:

- Classical and ML methods
- Good explanation of the different levels of automation (navigation, guidance and control)
- RL, IL, Safe reinforcement learning, multi-agent

[Automatic ship collision avoidance using deep reinforcement learning with LSTM in continuous action spaces](https://link.springer.com/article/10.1007/s00773-020-00755-0) **Quite close to what I do**:

- Bumper model that could be useful to me
- OZT (used for collision risk assessment) that extends CPA (closest point of approach) + inside OZT
- Bow crossing range
- Grid sensor for detecting OZT (kind of like my radar image)

[Automatic Collision Avoidance Using Deep Reinforcement Learning with Grid Sensor](https://link.springer.com/chapter/10.1007/978-3-030-37442-6_3)
- Similar to previous
- Has hyperparameters

[Collision Avoidance Road Test for COLREGS-Constrained Autonomous Vehicles](http://marinerobotics.mit.edu/sites/default/files/colregs.pdf)

[Evaluating of Marine Traffic Simulation System through Imazu Problem](http://www.naoe.eng.osaka-u.ac.jp/~hase/cv/papers3/166CAi2013A-GS12-1.pdf)

https://dl.acm.org/doi/10.1145/2903220.2903254
file:///tmp/mozilla_sandsj@sirehna.com0/jmse-07-00438-v3.pdf
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjG5qHmxMvxAhWwyYUKHXCqCgIQFjABegQIBBAD&url=https%3A%2F%2Fwww.revistas.usp.br%2Fmecatrone%2Farticle%2Fdownload%2F151953%2F149871%2F327550&usg=AOvVaw3ZVmbUxn9jNNXVv99PnUBG
file:///tmp/mozilla_sandsj@sirehna.com0/151953-Texto%20do%20artigo-327550-1-10-20181229.pdf
https://www.researchgate.net/requests/r90571115

https://arxiv.org/pdf/2101.00186.pdf
https://arxiv.org/pdf/2009.14551.pdf