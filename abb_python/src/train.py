import rospy
import atexit
import ray
from ray.rllib.agents.ddpg.apex import ApexDDPGTrainer
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.logger import UnifiedLogger
import gym
import os
import sys
import signal
from models import Q_KOMP
import abb_reach

num_of_workers = 2
horizon = 40

def cleanup():
    for worker in range(num_of_workers):
        file = open("gazebo_process_" + str(worker + 1) + ".txt", "r")
        gazebo_pgid = int(file.readline())
        file.close()
        file = open("moveit_process_" + str(worker + 1) + ".txt", "r")
        moveit_pgid = int(file.readline())
        file.close()
        print("Killing worker "+str(worker)+" with "+str(gazebo_pgid))
        os.killpg(gazebo_pgid, signal.SIGTERM)
        print("Killing worker "+str(worker)+" with "+str(moveit_pgid))
        os.killpg(moveit_pgid, signal.SIGTERM)

atexit.register(cleanup)
#ray.init(plasma_directory='/tmp/plasma')
#ray.init(object_store_memory=20**9)
#ray.init(redis_address="192.168.1.102:15346")
ray.init()
def on_episode_start(info):
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))



def on_episode_end(info):
    episode = info["episode"]
    print(episode.episode_id, episode.length)


def on_sample_end(info):
    print("returned sample batch of size {}".format(info["samples"].count))


def on_train_result(info):
    print("trainer.train() result: {} -> {} episodes".format(
        info["trainer"], info["result"]["episodes_this_iter"]))
    # you can mutate the result dict to add new fields to return
    info["result"]["callback_ok"] = True


def on_postprocess_traj(info):
    episode = info["episode"]
    batch = info["batch"]
    print("postprocessed {} steps".format(batch.count))
    if "num_batches" not in episode.custom_metrics:
        episode.custom_metrics["num_batches"] = 0
    episode.custom_metrics["num_batches"] += 1

#ApexDDPGTrainer._policy_graph = Q_KOMP
'''agent = DDPGAgent(config={"model": {
                "custom_preprocessor": "OurPreprocessor",
                         }, }, env=abb_reach.ABBReachEnv)
'''
'''
agent = ApexDDPGTrainer(config={"horizon":20,"callbacks": {
                "on_episode_start": tune.function(on_episode_start),
                "on_episode_step": None,
                "on_episode_end": tune.function(on_episode_end),
                "on_sample_end": tune.function(on_sample_end),
                "on_train_result": tune.function(on_train_result),
                "on_postprocess_traj": tune.function(on_postprocess_traj),
            }}, env=abb_reach.ABBReachEnv, )

#"observation_filter":"MeanStdFilter"
agent = ApexDDPGTrainer(config={"compress_observations": True, "learning_starts":500000, "parameter_noise": True, "horizon": horizon, "num_workers": num_of_workers,"train_batch_size": 32, #"num_cpus_per_worker": 2,
                "callbacks": {
                #"on_episode_start": tune.function(on_episode_start),
                #"on_episode_step": None,
                "on_episode_end": tune.function(on_episode_end),
                "on_sample_end": tune.function(on_sample_end),
                "on_train_result": tune.function(on_train_result),
                "on_postprocess_traj": tune.function(on_postprocess_traj),
            }}, env=abb_reach.ABBReachEnv)

while(True):
  agent.train()
'''
run_experiments({
        "Our_results": {
            "run": "APEX_DDPG",
            "env": abb_reach.ABBReachEnv, 
            #"resources_per_trial":{"cpu": 12, "gpu":1},
            "stop": {"timesteps_total": 100000,},
            "loggers":[UnifiedLogger],
            "config": {
                "compress_observations": True, "num_envs_per_worker": 1, "train_batch_size": 32, #"timesteps_per_iteration": 70, "min_iter_time_s":5,
                "horizon":horizon, "learning_starts":250000, "parameter_noise": True, "num_workers": num_of_workers, #"collect_metrics_timeout":960,
                "buffer_size":50000, "lr": 1e-2,  # to try different learning rates #"per_worker_exploration": True,
                "num_cpus_per_worker": 3, "num_gpus":1, "num_cpus_for_driver":3,
                "callbacks": {
                #"on_episode_start": tune.function(on_episode_start),
                #"on_episode_step": None,
                "on_episode_end": tune.function(on_episode_end),
                "on_sample_end": tune.function(on_sample_end),
                "on_train_result": tune.function(on_train_result),
                "on_postprocess_traj": tune.function(on_postprocess_traj),
                }
                },
        },
    })

