import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import optuna
from optuna.samplers import TPESampler

env = gym.make("ALE/SpaceInvaders-v5")
eval_env = Monitor(gym.make("ALE/SpaceInvaders-v5"))

def run_training(params, verbose=0, save_model=False):
    """
    model = DQN(
        policy='CnnPolicy', 
        env=env, 
        n_steps=1024,
        batch_size=64, 
        n_epochs=params['n_epochs'], # We're tuning this.
        gamma=params['gamma'], # We're tuning this.
        gae_lambda=0.98, 
        ent_coef=0.01, 
        verbose=verbose
    )
    """
    model = DQN(
        policy='CnnPolicy', 
        env=env, 
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts = 100000,
        batch_size =32,
        train_freq=4,
        gradient_steps=1,
        optimize_memory_usage=True,
        target_update_interval=1000,
        exploration_fraction=0.025,
        exploration_final_eps=0.01
    )

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    score = mean_reward - std_reward

    if save_model:
        model.save("SpacesInvadersNoFrameskip-v4")

    return model, score

def objective(trial):
  params = {
      "exploration_fraction": trial.suggest_uniform("exploration_fraction", 0.0025,0.99), 
      "learning_rate": trial.suggest_uniform("learning_rate", 0.0001, 0.9999)
  }
  model, score = run_training(params)
  return score

study = optuna.create_study(sampler=TPESampler(), study_name="DQN-SpacesInvadersNoFrameskip-v4", direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial score:", study.best_trial.values)
print("Best trial hyperparameters:", study.best_trial.params)