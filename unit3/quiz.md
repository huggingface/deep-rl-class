# Knowledge Check ✔️

The best way to learn and [avoid the illusion of competence](https://fr.coursera.org/lecture/learning-how-to-learn/illusions-of-competence-BuFzf) **is to test yourself.** This will help you to find **where you need to reinforce your knowledge**. 

📝 Take a piece of paper and try to answer by writing, **then check the solutions**.

### Q1: What are tabular methods?

<details>
<summary>Solution</summary>
  
*Tabular methods* are a type of problems in which the state and actions spaces are small enough to approximate value functions to be **represented as arrays and tables**. For instance, **Q-Learning is a tabular method** since we use a table to represent the state,action value pairs.
  
📖 If you don't remember, check 👉 https://huggingface.co/blog/deep-rl-dqn#from-q-learning-to-deep-q-learning
  
</details>


### Q2: Why we can't use a classical Q-Learning to solve an Atari Game?
<details>
<summary>Solution</summary>
  
Atari environments have an observation space with a shape of (210, 160, 3), containing values ranging from 0 to 255, so that gives us 256^(210x160x3) = 256^100800 (**for comparison, we have approximately 10^80 atoms in the observable universe**).
  
Therefore, the state space is gigantic; hence creating and updating a Q-table for that environment **would not be efficient**. In this case, the best idea is to approximate the Q-values instead of a Q-table using a parametrized Q-function $Q_{\theta}(s,a)$.

📖 If you don't remember, check 👉 https://huggingface.co/blog/deep-rl-dqn#from-q-learning-to-deep-q-learning
</details>

### Q3: Why do we stack four frames together when we use frames as input in Deep Q-Learning?

<details>
<summary>Solution</summary>
  
We stack frames together because it helps us **handle the problem of temporal limitation**. Since one frame is not enough to capture temporal information.
For instance, in pong, our agent **will be unable to know the ball direction if it gets only one frame**.
  
<img src="https://huggingface.co/blog/assets/78_deep_rl_dqn/temporal-limitation.jpg" alt="Temporal limitation"/>
<img src="https://huggingface.co/blog/assets/78_deep_rl_dqn/temporal-limitation-2.jpg" alt="Temporal limitation"/>

  
📖 If you don't remember, check 👉 https://huggingface.co/blog/deep-rl-dqn#preprocessing-the-input-and-temporal-limitation
</details>

### Q4: What are the two phases of Deep Q-Learning?

<details>
 <summary>Solution</summary>
  
The Deep Q-Learning training algorithm has two phases:
- *Sampling* : we perform actions and **store the observed experiences tuples in a replay memory**.
- *Training* : Select the small batch of tuple randomly and **learn from it using a gradient descent update step**.

📖 If you don't remember, check 👉 [https://huggingface.co/blog/deep-rl-dqn#preprocessing-the-input-and-temporal-limitation](https://huggingface.co/blog/deep-rl-dqn#the-deep-q-learning-algorithm)
</details>

### Q5: Why do we create a replay memory in Deep Q-Learning?

<details>
   <summary>Solution</summary>
  
**1. Make more efficient use of the experiences during the training**

Usually, in online reinforcement learning, we interact in the environment, get experiences (state, action, reward, and next state), learn from them (update the neural network) and discard them.
But with experience replay, **we create a replay buffer that saves experience samples that we can reuse during the training**.
  
**2. Avoid forgetting previous experiences and reduce the correlation between experiences**
  
  The problem we get if we give sequential samples of experiences to our neural network is that it **tends to forget the previous experiences as it overwrites new experiences**. For instance, if we are in the first level and then the second, which is different, our agent can forget how to behave and play in the first level.

📖 If you don't remember, check 👉 https://huggingface.co/blog/deep-rl-dqn#experience-replay-to-make-more-efficient-use-of-experiences
  
</details>

### Q6: How do we use Double Deep Q-Learning?


<details>
  <summary>Solution</summary>
  
  When we compute the Q target, we use two networks to decouple the action selection from the target Q value generation. We:
  
  - Use our *DQN network* to **select the best action to take for the next state** (the action with the highest Q value).
  
  - Use our *Target network* to calculate **the target Q value of taking that action at the next state**.
  
</details>

---

Congrats on **finishing this Quiz** 🥳, if you missed some elements, take time to [read the chapter again](https://huggingface.co/blog/deep-rl-dqn) to reinforce (😏) your knowledge.

**Keep Learning, Stay Awesome**
