# Introducing the Clipped Surrogate Objective Function
## Recap: The Policy Objective Function

Let’s remember what the objective is to optimize in Reinforce:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/lpg.jpg" alt="Reinforce"/>

The idea was that by taking a gradient ascent step on this function (equivalent to taking gradient descent of the negative of this function), we would **push our agent to take actions that lead to higher rewards and avoid harmful actions.**

However, the problem comes from the step size:
- Too small, **the training process was too slow**
- Too high, **there was too much variability in the training**

With PPO, the idea is to constrain our policy update with a new objective function called the *Clipped surrogate objective function* that **will constrain the policy change in a small range using a clip.**

This new function **is designed to avoid destructively large weights updates** :

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/ppo-surrogate.jpg" alt="PPO surrogate function"/>

Let’s study each part to understand how it works.

## The Ratio Function
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/ratio1.jpg" alt="Ratio"/>

This ratio is calculated as follows:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/ratio2.jpg" alt="Ratio"/>

It’s the probability of taking action \\( a_t \\) at state \\( s_t \\) in the current policy, divided by the same for the previous policy.

As we can see, \\( r_t(\theta) \\) denotes the probability ratio between the current and old policy:

- If \\( r_t(\theta) > 1 \\), the **action \\( a_t \\) at state \\( s_t \\) is more likely in the current policy than the old policy.**
- If \\( r_t(\theta) \\) is between 0 and 1, the **action is less likely for the current policy than for the old one**.

So this probability ratio is an **easy way to estimate the divergence between old and current policy.**

## The unclipped part of the Clipped Surrogate Objective function
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/unclipped1.jpg" alt="PPO"/>

This ratio **can replace the log probability we use in the policy objective function**. This gives us the left part of the new objective function: multiplying the ratio by the advantage.
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/unclipped2.jpg" alt="PPO"/>
  <figcaption><a href="https://arxiv.org/pdf/1707.06347.pdf">Proximal Policy Optimization Algorithms</a></figcaption>
</figure>

However, without a constraint, if the action taken is much more probable in our current policy than in our former, **this would lead to a significant policy gradient step** and, therefore, an **excessive policy update.**

## The clipped Part of the Clipped Surrogate Objective function

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/clipped.jpg" alt="PPO"/>

Consequently, we need to constrain this objective function by penalizing changes that lead to a ratio far away from 1 (in the paper, the ratio can only vary from 0.8 to 1.2).

**By clipping the ratio, we ensure that we do not have a too large policy update because the current policy can't be too different from the older one.**

To do that, we have two solutions:

- *TRPO (Trust Region Policy Optimization)* uses KL divergence constraints outside the objective function to constrain the policy update. But this method **is complicated to implement and takes more computation time.**
- *PPO* clip probability ratio directly in the objective function with its **Clipped surrogate objective function.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/clipped.jpg" alt="PPO"/>

This clipped part is a version where \\( r_t(\theta) \\) is clipped between  \\( [1 - \epsilon, 1 + \epsilon] \\).

With the Clipped Surrogate Objective function, we have two probability ratios, one non-clipped and one clipped in a range between  \\( [1 - \epsilon, 1 + \epsilon] \\), epsilon is a hyperparameter that helps us to define this clip range (in the paper  \\( \epsilon = 0.2 \\).).

Then, we take the minimum of the clipped and non-clipped objective, **so the final objective is a lower bound (pessimistic bound) of the unclipped objective.**

Taking the minimum of the clipped and non-clipped objective means **we'll select either the clipped or the non-clipped objective based on the ratio and advantage situation**.
