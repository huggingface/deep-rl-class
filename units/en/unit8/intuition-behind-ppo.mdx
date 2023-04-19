# The intuition behind PPO [[the-intuition-behind-ppo]]


The idea with Proximal Policy Optimization (PPO) is that we want to improve the training stability of the policy by limiting the change you make to the policy at each training epoch: **we want to avoid having too large of a policy update.**

For two reasons:
- We know empirically that smaller policy updates during training are **more likely to converge to an optimal solution.**
- A too-big step in a policy update can result in falling “off the cliff” (getting a bad policy) **and taking a long time or even having no possibility to recover.**

<figure class="image table text-center m-0 w-full">
  <img class="center" src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/cliff.jpg" alt="Policy Update cliff"/>
  <figcaption>Taking smaller policy updates to improve the training stability</figcaption>
  <figcaption>Modified version from RL — Proximal Policy Optimization (PPO) <a href="https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12">Explained by Jonathan Hui</a></figcaption>
</figure>

**So with PPO, we update the policy conservatively**. To do so, we need to measure how much the current policy changed compared to the former one using a ratio calculation between the current and former policy. And we clip this ratio in a range \\( [1 - \epsilon, 1 + \epsilon] \\), meaning that we **remove the incentive for the current policy to go too far from the old one (hence the proximal policy term).**
