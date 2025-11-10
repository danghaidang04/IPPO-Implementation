# Algs/IPPO/Ippo_config.py

IPPO_HYPERPARAMS = {
    "lr": 1e-4,                 # Learning rate
    "gamma": 0.99,              # Discount factor
    "lambda": 1.0,              # GAE parameter
    "clip_param": 0.2,          # PPO clip parameter
    "vf_clip_param": 10.0,      # Value function clip
    "entropy_coeff": 0.01,      # Hệ số entropy (khuyến khích khám phá)
    "num_sgd_iter": 10,         # Số lần lặp lại SGD
    "sgd_minibatch_size": 128,  # Kích thước minibatch
    "train_batch_size": 2000,   # Kích thước batch thu thập từ các worker
}