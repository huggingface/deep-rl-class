# The SnowballTarget Environment

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballtarget.gif" alt="SnowballTarget"/>

SnowballTarget is an environment we created at Hugging Face using assets from [Kay Lousberg](https://kaylousberg.com/). We have an optional section at the end of this Unit **if you want to learn to use Unity and create your environments**.

## The agent's Goal

The first agent you're going to train is called Julien the bear 🐻. Julien is trained **to hit targets with snowballs**.

The Goal in this environment is that Julien **hits as many targets as possible in the limited time** (1000 timesteps). It will need **to place itself correctly in relation to the target and shoot**to do that.

In addition, to avoid "snowball spamming" (aka shooting a snowball every timestep), **Julien has a "cool off" system** (it needs to wait 0.5 seconds after a shoot to be able to shoot again).

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/cooloffsystem.gif" alt="Cool Off System"/>
<figcaption>The agent needs to wait 0.5s before being able to shoot a snowball again</figcaption>
</figure>

## The reward function and the reward engineering problem

The reward function is simple. **The environment gives a +1 reward every time the agent's snowball hits a target**. Because the agent's Goal is to maximize the expected cumulative reward, **it will try to hit as many targets as possible**.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballtarget_reward.png" alt="Reward system"/>

We could have a more complex reward function (with a penalty to push the agent to go faster, for example). But when you design an environment, you need to avoid the *reward engineering problem*, which is having a too complex reward function to force your agent to behave as you want it to do.
Why? Because by doing that, **you might miss interesting strategies that the agent will find with a simpler reward function**.

In terms of code, it looks like this:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballtarget-reward-code.png" alt="Reward"/>


## The observation space

Regarding observations, we don't use normal vision (frame), but **we use raycasts**.

Think of raycasts as lasers that will detect if they pass through an object.

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit5/raycasts.png" alt="Raycasts"/>
<figcaption>Source: <a href="https://github.com/Unity-Technologies/ml-agents">ML-Agents documentation</a></figcaption>
</figure>


In this environment, our agent has multiple set of raycasts:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowball_target_raycasts.png" alt="Raycasts"/>

In addition to raycasts, the agent gets a "can I shoot" bool as observation.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballtarget-obs-code.png" alt="Obs"/>

## The action space

The action space is discrete:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballtarget_action_space.png" alt="Action Space"/>
