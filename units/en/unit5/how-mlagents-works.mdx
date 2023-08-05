# How do Unity ML-Agents work? [[how-mlagents-works]]

Before training our agent, we need to understand **what ML-Agents is and how it works**.

## What is Unity ML-Agents? [[what-is-mlagents]]

[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) is a toolkit for the game engine Unity that **allows us to create environments using Unity or use pre-made environments to train our agents**.

It’s developed by [Unity Technologies](https://unity.com/), the developers of Unity, one of the most famous Game Engines used by the creators of Firewatch, Cuphead, and Cities: Skylines.

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit5/firewatch.jpeg" alt="Firewatch"/>
<figcaption>Firewatch was made with Unity</figcaption>
</figure>

## The six components [[six-components]]

With Unity ML-Agents, you have six essential components:

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit5/mlagents-1.png" alt="MLAgents"/>
<figcaption>Source: <a href="https://unity-technologies.github.io/ml-agents/">Unity ML-Agents Documentation</a> </figcaption>
</figure>

- The first is the *Learning Environment*, which contains **the Unity scene (the environment) and the environment elements** (game characters).
- The second is the *Python Low-level API*, which contains **the low-level Python interface for interacting and manipulating the environment**. It’s the API we use to launch the training.
- Then, we have the *External Communicator* that **connects the Learning Environment (made with C#) with the low level Python API (Python)**.
- The *Python trainers*: the **Reinforcement algorithms made with PyTorch (PPO, SAC…)**.
- The *Gym wrapper*: to encapsulate the RL environment in a gym wrapper.
- The *PettingZoo wrapper*: PettingZoo is the multi-agents version of the gym wrapper.

## Inside the Learning Component [[inside-learning-component]]

Inside the Learning Component, we have **two important elements**:

- The first is the *agent component*, the actor of the scene. We’ll **train the agent by optimizing its policy** (which will tell us what action to take in each state). The policy is called the *Brain*.
- Finally, there is the *Academy*. This component **orchestrates agents and their decision-making processes**. Think of this Academy as a teacher who handles Python API requests.

To better understand its role, let’s remember the RL process. This can be modeled as a loop that works like this:

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process.jpg" alt="The RL process" width="100%">
<figcaption>The RL Process: a loop of state, action, reward and next state</figcaption>
<figcaption>Source: <a href="http://incompleteideas.net/book/RLbook2020.pdf">Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto</a></figcaption>
</figure>

Now, let’s imagine an agent learning to play a platform game. The RL process looks like this:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process_game.jpg" alt="The RL process" width="100%">

- Our Agent receives **state  \\(S_0\\)** from the **Environment** — we receive the first frame of our game (Environment).
- Based on that **state \\(S_0\\),** the Agent takes **action \\(A_0\\)** — our Agent will move to the right.
- The environment goes to a **new** **state \\(S_1\\)** — new frame.
- The environment gives some **reward \\(R_1\\)** to the Agent — we’re not dead *(Positive Reward +1)*.

This RL loop outputs a sequence of **state, action, reward and next state.** The goal of the agent is to **maximize the expected cumulative reward**.

The Academy will be the one that will **send the order to our Agents and ensure that agents are in sync**:

- Collect Observations
- Select your action using your policy
- Take the Action
- Reset if you reached the max step or if you’re done.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit5/academy.png" alt="The MLAgents Academy" width="100%">


Now that we understand how ML-Agents works, **we’re ready to train our agents.**
