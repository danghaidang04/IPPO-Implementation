# file: ippo_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Giả định actor.py và critic.py nằm cùng thư mục
from networks.actor import Actor
from networks.critic import Critic

class RolloutBuffer:
    """Một buffer đơn giản để lưu trữ transitions cho TẤT CẢ agents."""
    def __init__(self):
        # Sử dụng dict để lưu trữ buffer riêng cho từng agent
        self.agent_buffers = {}

    def add_agent(self, agent_id):
        """Thêm một agent mới vào buffer."""
        if agent_id not in self.agent_buffers:
            self.agent_buffers[agent_id] = {
                "states": [],
                "actions": [],
                "logprobs": [],
                "rewards": [],
                "dones": []
            }

    def push(self, agent_id, state, action, logprob, reward, done):
        """Đẩy một transition vào buffer của agent tương ứng."""
        if agent_id in self.agent_buffers:
            buffer = self.agent_buffers[agent_id]
            buffer["states"].append(state)
            buffer["actions"].append(action)
            buffer["logprobs"].append(logprob)
            buffer["rewards"].append(reward)
            buffer["dones"].append(done)

    def get_data(self, agent_id):
        """Lấy toàn bộ dữ liệu của một agent và chuyển sang tensor."""
        buffer = self.agent_buffers[agent_id]
        return {
            "states": torch.tensor(np.array(buffer["states"]), dtype=torch.float32),
            "actions": torch.tensor(buffer["actions"], dtype=torch.int64),
            "logprobs": torch.tensor(buffer["logprobs"], dtype=torch.float32),
            "rewards": torch.tensor(buffer["rewards"], dtype=torch.float32),
            "dones": torch.tensor(buffer["dones"], dtype=torch.float32)
        }
        
    def clear(self):
        """Xóa tất cả dữ liệu trong buffer."""
        self.agent_buffers = {}


class IPPOTrainer:
    def __init__(self, env, actor_lr, critic_lr, gamma, gae_lambda,
                 ppo_clip, k_epochs, entropy_coeff, use_gpu=True):
        
        self.env = env
        self.agent_ids = self.env.possible_agents
        
        # Thiết lập Device (GPU/CPU)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")

        # Lấy thông tin state/action (giả định đồng nhất)
        # Lấy state và action space của agent đầu tiên làm đại diện
        self.state_dim = self.env.observation_space(self.agent_ids[0]).shape[0]
        self.action_dim = self.env.action_space(self.agent_ids[0]).n
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.k_epochs = k_epochs
        self.entropy_coeff = entropy_coeff

        # --- Khởi tạo mạng và optimizer cho MỖI agent ---
        self.actors = {}
        self.critics = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}

        for agent_id in self.agent_ids:
            # Tạo Actor
            actor = Actor(self.state_dim, self.action_dim).to(self.device)
            self.actors[agent_id] = actor
            self.actor_optimizers[agent_id] = optim.Adam(actor.parameters(), lr=actor_lr)
            
            # Tạo Critic
            critic = Critic(self.state_dim).to(self.device)
            self.critics[agent_id] = critic
            self.critic_optimizers[agent_id] = optim.Adam(critic.parameters(), lr=critic_lr)

        # Buffer
        self.buffer = RolloutBuffer()
        
        # Loss function
        self.mse_loss = nn.MSELoss()

    def _compute_advantages_gae(self, agent_id, data):
        """Tính GAE (Generalized Advantage Estimation) cho một agent."""
        rewards = data["rewards"]
        dones = data["dones"]
        
        with torch.no_grad():
            states = data["states"].to(self.device)
            values = self.critics[agent_id](states).squeeze()

        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        
        gae = 0
        last_value = 0 # Giả sử V(s_cuối) = 0 nếu done
        
        # Tính GAE từ cuối về đầu
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t] # 0 nếu done, 1 nếu không
            
            # Ước tính V(s_next)
            # Nếu là bước cuối cùng (t == len(rewards) - 1), dùng last_value (thường là 0)
            # Nếu không, dùng value[t+1]
            next_value = values[t+1] if t < len(rewards) - 1 else last_value
            
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            
            # Return = Advantage + Value
            returns[t] = gae + values[t]
            advantages[t] = gae

        # Chuẩn hóa Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.cpu(), returns.cpu()

    def learn(self, num_rollout_steps):
        """
        Hàm chính: 
        1. Thu thập dữ liệu (rollout)
        2. Cập nhật mạng (update)
        """
        
        # --- 1. Thu thập dữ liệu (Rollout) ---
        self.buffer.clear() # Xóa buffer cũ
        for agent_id in self.agent_ids:
            self.buffer.add_agent(agent_id) # Khởi tạo lại buffer cho agents
            
        obs, _ = self.env.reset()
        
        for _ in range(num_rollout_steps):
            # Nếu env đã kết thúc, reset
            if not self.env.agents:
                obs, _ = self.env.reset()
                for agent_id in self.agent_ids:
                    self.buffer.add_agent(agent_id)
            
            # Chuyển state sang tensor và đưa lên device
            current_obs_tensors = {
                agent: torch.tensor(o, dtype=torch.float32).to(self.device)
                for agent, o in obs.items()
            }
            
            actions = {}
            logprobs = {}
            
            # Lấy hành động cho từng agent
            with torch.no_grad():
                for agent_id, state_tensor in current_obs_tensors.items():
                    action, logprob = self.actors[agent_id].get_action(state_tensor)
                    actions[agent_id] = action
                    logprobs[agent_id] = logprob.cpu() # Lưu logprob

            # Thực thi hành động trong env
            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
            
            # Xử lý dones (terminated hoặc truncated)
            dones = {agent: term or trunc for agent, (term, trunc) in 
                     zip(terminations, zip(terminations.values(), truncations.values()))}

            # Lưu trữ transitions vào buffer
            # 'obs' chứa state của agent *trước khi* thực hiện hành động
            for agent_id in obs.keys():
                self.buffer.push(
                    agent_id,
                    obs[agent_id],
                    actions[agent_id],
                    logprobs[agent_id],
                    rewards[agent_id],
                    dones[agent_id]
                )

            # Cập nhật state tiếp theo
            obs = next_obs
        
        # --- 2. Cập nhật mạng (Update) ---
        self._update_policies()

    def _update_policies(self):
        """Cập nhật policy cho từng agent một cách độc lập."""
        
        for agent_id in self.agent_ids:
            # Lấy dữ liệu và tính GAE
            try:
                data = self.buffer.get_data(agent_id)
            except KeyError:
                continue # Agent này không có dữ liệu (có thể đã done sớm)

            if len(data["states"]) == 0:
                continue # Bỏ qua nếu không có dữ liệu

            advantages, returns = self._compute_advantages_gae(agent_id, data)
            
            # Đưa dữ liệu lên device
            states = data["states"].to(self.device)
            actions = data["actions"].to(self.device)
            old_logprobs = data["logprobs"].to(self.device)
            advantages = advantages.to(self.device)
            returns = returns.to(self.device)

            # --- Vòng lặp cập nhật K-epochs (PPO) ---
            for _ in range(self.k_epochs):
                
                # --- Cập nhật Actor (Policy Loss) ---
                new_logprobs, dist_entropy = self.actors[agent_id].evaluate(states, actions)
                
                # Tính tỷ lệ (ratio)
                ratio = torch.exp(new_logprobs - old_logprobs)
                
                # Tính PPO-Clip loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
                
                # Loss = -min(surr1, surr2) - (entropy bonus)
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * dist_entropy.mean()
                
                # Cập nhật Actor
                self.actor_optimizers[agent_id].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[agent_id].step()

                # --- Cập nhật Critic (Value Loss) ---
                # Lấy V(s) mới từ critic
                new_values = self.critics[agent_id](states).squeeze()
                
                # Critic loss = MSE(V_mới, Returns)
                # (Chúng ta cũng có thể dùng PPO value clipping, nhưng MSE đơn giản và hiệu quả)
                critic_loss = self.mse_loss(new_values, returns)
                
                # Cập nhật Critic
                self.critic_optimizers[agent_id].zero_grad()
                critic_loss.backward()
                self.critic_optimizers[agent_id].step()

    def save_models(self, path_prefix):
        """Lưu mô hình Actor và Critic cho từng agent."""
        for agent_id in self.agent_ids:
            torch.save(self.actors[agent_id].state_dict(), f"{path_prefix}_actor_{agent_id}.pth")
            torch.save(self.critics[agent_id].state_dict(), f"{path_prefix}_critic_{agent_id}.pth")


    def load_models(self, path_prefix):
        """Tải mô hình Actor và Critic cho từng agent."""
        for agent_id in self.agent_ids:
            self.actors[agent_id].load_state_dict(torch.load(f"{path_prefix}_actor_{agent_id}.pth"))
            self.critics[agent_id].load_state_dict(torch.load(f"{path_prefix}_critic_{agent_id}.pth"))

            