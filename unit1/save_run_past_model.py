import gym
from stable_baselines3 import PPO
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

#if there is no any saved model, directly begin traning.
#if there is a saved model ask which model you want to use make further tranining.

models_dir = "models/PPO"
logdir = "logs"
env = make_vec_env('LunarLander-v2', n_envs=16)

model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 1024,
        batch_size = 64,
        n_epochs = 4,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        tensorboard_log=logdir,
        verbose=1)

def train_model(timesteps=1000, iter=10, saved_model_fn_int=0):
    #saved_model_fn_int: this parameter is for using past model.
    TIMESTEPS = timesteps
    for i in range(iter):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        saved_model_file_name = saved_model_fn_int + i * TIMESTEPS
        model.save(f"{models_dir}/{saved_model_file_name}")

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    train_model(timesteps=1000,iter=30)

prefer = input("Do you want to use past model?(y/n): ").strip()
if prefer == "n":
    train_model(timesteps=1000,iter=30)
else:
    saved_model = int(input("Write the name of the model you want to use ?(can be found in models/,without .zip): ").replace('.zip','').strip())
    model_path = f"{models_dir}/{saved_model}.zip"
    model = PPO.load(model_path, env=env)
    train_model(timesteps=1000, iter=30, saved_model_fn_int=saved_model)