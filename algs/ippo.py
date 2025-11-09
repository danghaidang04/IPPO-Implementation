import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from envs.simple_env import env as make_env

def create_env():
    """Return a PettingZoo wrapped environment"""
    return PettingZooEnv(make_env())

def build_policies_from_env():
    """Return policies dict và policy mapping function"""
    env_instance = make_env()
    env_instance.reset()
    agents = env_instance.possible_agents

    # Lấy observation & action space của agent đầu tiên
    obs_space = env_instance.observation_spaces[agents[0]]
    act_space = env_instance.action_spaces[agents[0]]

    # Tạo policies dict
    policies = {agent: (None, obs_space, act_space, {}) for agent in agents}

    # Mapping function: mỗi agent dùng policy riêng
    policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id
    return policies, policy_mapping_fn

def train_ippo(config=None):
    config = config or {}

    # 3.1 Khởi động Ray
    ray.init(ignore_reinit_error=True)

    # 3.2 Đăng ký environment
    register_env("simple_ma_env", create_env)

    # 3.3 Build policies
    policies, mapping_fn = build_policies_from_env()

    # 3.4 Config PPO
    ppo_config = {
        "env": "simple_ma_env",
        "framework": "torch",
        "num_workers": 1,
        "train_batch_size": 2000,
        "rollout_fragment_length": 100,
        "num_sgd_iter": 10,
        "lr": 1e-3,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": mapping_fn,
        },
        "log_level": "INFO",
    }
    ppo_config.update(config)

    # 3.5 Tạo trainer
    trainer = PPO(config=ppo_config)

    # 3.6 Training loop
    print("🚀 Start training IPPO...")
    for i in range(30):
        result = trainer.train()
        print(f"Iter {i} | Reward_mean: {result.get('episode_reward_mean', 0):.3f}")
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("Checkpoint saved:", checkpoint)

    ray.shutdown()