# Unit 8: Proximal Policy Optimization (PPO) with PyTorch

Today we'll learn about Proximal Policy Optimization (PPO), an architecture that improves our agent's training stability by avoiding too large policy updates. To do that, we use a ratio that will indicates the difference between our current and old policy and clip this ratio from a specific range  $[1 - \epsilon, 1 + \epsilon]$. Doing this will ensure that our policy update will not be too large and that the training is more stable.

And then, after the theory, we'll code a PPO architecture from scratch using PyTorch and bulletproof our implementation with CartPole-v1 and LunarLander-v2.

🏆 You'll then be able to **compare your agent’s results with other classmates thanks to a leaderboard** 🔥 👉 https://huggingface.co/spaces/chrisjay/Deep-Reinforcement-Learning-Leaderboard

<img src="assets/img/LunarLander.gif" alt="LunarLander"/>

Let's get started 🥳

## Required time ⏱️
The required time for this unit is, approximately:
- 1 hour for the theory.
- 2 hours for the hands-on.

## Start this Unit 🚀
Here are the steps for this Unit:

1️⃣ 📖 [Read Proximal Policy Optimization Chapter](https://huggingface.co/blog/deep-rl-ppo).

2️⃣ 👩‍💻 Then dive on the hands-on:

The hands-on 👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit8/unit8.ipynb)

Thanks to a leaderboard, you'll be able to compare your results with other classmates and exchange the best practices to improve your agent's scores Who will win the challenge for Unit 8 🏆?

The leaderboard 👉 https://huggingface.co/spaces/chrisjay/Deep-Reinforcement-Learning-Leaderboard

## Additional readings 📚
- [Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization by Daniel Bick](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf) 
- [What is the way to understand Proximal Policy Optimization Algorithm in RL?](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl)
- [Foundations of Deep RL Series, L4 TRPO and PPO by Pieter Abbeel](https://youtu.be/KjWF8VIMGiY)
- [OpenAI PPO Blogpost](https://openai.com/blog/openai-baselines-ppo/)
- [Spinning Up RL PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Paper Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [The 37 Implementation Details of Proximal Policy Optimization](https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/)
- [Importance Sampling Explained](https://youtu.be/C3p2wI4RAi8)

## How to make the most of this course

To make the most of the course, my advice is to:

- **Participate in Discord** and join a study group.
- **Read multiple times** the theory part and takes some notes
- Don’t just do the colab. When you learn something, try to change the environment, change the parameters and read the libraries' documentation. Have fun 🥳
- Struggling is **a good thing in learning**. It means that you start to build new skills. Deep RL is a complex topic and it takes time to understand. Try different approaches, use our additional readings, and exchange with classmates on discord.

## This is a course built with you 👷🏿‍♀️

We want to improve and update the course iteratively with your feedback. **If you have some, please fill this form** 👉 https://forms.gle/3HgA7bEHwAmmLfwh9

## Don’t forget to join the Community 📢

We have a discord server where you **can exchange with the community and with us, create study groups to grow each other and more** 

👉🏻 [https://discord.gg/aYka4Yhff9](https://discord.gg/aYka4Yhff9).

Don’t forget to **introduce yourself when you sign up 🤗**

❓If you have other questions, [please check our FAQ](https://github.com/huggingface/deep-rl-class#faq)

### Keep learning, stay awesome 🤗
