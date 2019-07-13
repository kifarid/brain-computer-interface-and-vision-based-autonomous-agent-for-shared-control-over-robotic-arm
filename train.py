##import rospy
import atexit
import ray
#from ray.rllib.agents.ddpg.apex import ApexDDPGTrainer
from ray import tune
from ray.tune import run_experiments, grid_search
import gym


num_of_workers = 12
horizon = 40

#ray.init()
#ray.init(redis_address="localhost:48842")
#ray.init(redis_address="localhost:14458",huge_pages= True, plasma_directory='/mnt/hugepages')
#ray.init(plasma_directory='/tmp/plasma')
ray.init(object_store_memory=175000000000, redis_max_memory=20000000000)


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


run_experiments({
        "Our_results": {
            "run": "APEX_DDPG",
            "env": 'FetchPickAndPlace-v1',
            "stop": {"episode_reward_mean": 20,},
            #"resources_per_trial":{"cpu": 10, "gpu": 1},
            "config": {
                "actor_hiddens": [256,256,256,128],
                "critic_hiddens": [256,256,256,128],
                "actor_hidden_activation": "relu",  "critic_hidden_activation": "relu",
                "n_step":2, 
                "compress_observations": True, "prioritized_replay": False,
                "timesteps_per_iteration": 1500, "min_iter_time_s":30,
                "horizon":horizon, "parameter_noise": True,
                "num_workers": 14, "collect_metrics_timeout":1600,
                "buffer_size":1000000, "lr": 1e-3,  # to try different learning rates #"per_worker_exploration": True,
                "num_cpus_per_worker": 0.5,
                "num_cpus_for_driver":1,
                "batch_mode": "complete_episodes",
                "per_worker_exploration" : True,
                "callbacks": {
                #"on_episode_start": tune.function(on_episode_start),
                #"on_episode_step": None,
                #"on_episode_end": tune.function(on_episode_end),
                #"on_sample_end": tune.function(on_sample_end),
                "on_train_result": tune.function(on_train_result),
                #"on_postprocess_traj": tune.function(on_postprocess_traj),
                }
                },
        },
    })
