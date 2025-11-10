import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv


class SimpleTagEnv(ParallelEnv):
    metadata = {
        "name": "simple_tag_v0",
    }

    def __init__(self, grid_size=10, max_steps=100):
        """
        Khởi tạo môi trường.
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Danh sách các agent có thể có trong môi trường
        self.possible_agents = ["agent_0", "agent_1"]
        # self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        # Lưu trữ vị trí của các agent
        self.agent_positions = {agent: None for agent in self.possible_agents}
        
        # Đếm số bước
        self.step_count = 0

    def reset(self, seed=None, options=None):
        """
        Reset môi trường về trạng thái ban đầu.
        """
        # Reset các agent còn "sống"
        self.agents = self.possible_agents[:]
        self.step_count = 0
        
        # Đặt vị trí ban đầu ngẫu nhiên (hoặc cố định)
        # agent_0 (chaser) ở góc (0, 0)
        self.agent_positions["agent_0"] = np.array([0, 0])
        # agent_1 (evader) ở góc (size-1, size-1)
        self.agent_positions["agent_1"] = np.array([self.grid_size - 1, self.grid_size - 1])
        
        # Lấy observations và infos ban đầu
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        """
        Thực hiện một bước trong môi trường.
        Input: actions - một dict {agent_id: action}
        Output: observations, rewards, terminations, truncations, infos
        """
        # 1. Cập nhật vị trí của các agent dựa trên hành động
        for agent, action in actions.items():
            self._move_agent(agent, action)
            
        self.step_count += 1

        # 2. Tính toán kết quả (bị bắt, hết giờ, hay tiếp tục)
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}

        chaser_pos = self.agent_positions["agent_0"]
        evader_pos = self.agent_positions["agent_1"]

        # Kiểm tra bị bắt (Termination)
        if np.array_equal(chaser_pos, evader_pos):
            terminations = {agent: True for agent in self.agents}
            rewards["agent_0"] = 10.0
            rewards["agent_1"] = -10.0
        
        # Kiểm tra hết giờ (Truncation)
        elif self.step_count >= self.max_steps:
            truncations = {agent: True for agent in self.agents}
            rewards["agent_0"] = 0.0  # Không bắt được
            rewards["agent_1"] = 5.0  # Thưởng vì sống sót
            
        # Trạng thái bình thường
        else:
            rewards["agent_0"] = -0.1 # Phí di chuyển
            rewards["agent_1"] = 0.1  # Thưởng vì còn sống

        # 3. Lấy observation mới
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        # Nếu một agent kết thúc (terminated/truncated), nó sẽ bị xóa khỏi danh sách self.agents
        # cho bước tiếp theo (PettingZoo tự động xử lý việc này)
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _move_agent(self, agent, action):
        """Hàm helper để di chuyển agent và xử lý va chạm tường."""
        pos = self.agent_positions[agent]
        
        if action == 0:  # Đứng yên
            pass
        elif action == 1:  # Lên
            pos[1] += 1
        elif action == 2:  # Xuống
            pos[1] -= 1
        elif action == 3:  # Trái
            pos[0] -= 1
        elif action == 4:  # Phải
            pos[0] += 1
            
        # Xử lý va chạm biên (clipping)
        pos = np.clip(pos, 0, self.grid_size - 1)
        self.agent_positions[agent] = pos

    def _get_obs(self):
        """
        Lấy observation cho tất cả agents.
        Mỗi agent thấy vị trí (x, y) của cả hai.
        Observation: [chaser_x, chaser_y, evader_x, evader_y]
        """
        obs_data = np.concatenate([
            self.agent_positions["agent_0"],
            self.agent_positions["agent_1"]
        ]).astype(np.float32)
        
        return {agent: obs_data for agent in self.agents}

    def observation_space(self, agent):
        """
        Trả về không gian quan sát của một agent.
        Box(low, high, shape)
        """
        return Box(
            low=0,
            high=self.grid_size - 1,
            shape=(4,),  # [x1, y1, x2, y2]
            dtype=np.float32
        )

    def action_space(self, agent):
        """
        Trả về không gian hành động của một agent.
        Discrete(n) - 0, 1, 2, 3, 4
        """
        return Discrete(5)

    def render(self):
        """
        Vẽ lại môi trường (dạng text-based).
        """
        # Tạo một lưới rỗng
        grid = np.full((self.grid_size, self.grid_size), '_')
        
        # Đặt vị trí evader (E)
        pos_e = self.agent_positions["agent_1"]
        grid[pos_e[1], pos_e[0]] = 'E'
        
        # Đặt vị trí chaser (C)
        pos_c = self.agent_positions["agent_0"]
        grid[pos_c[1], pos_c[0]] = 'C'
        
        # Nếu trùng vị trí
        if np.array_equal(pos_c, pos_e):
            grid[pos_c[1], pos_c[0]] = 'X' # Bị bắt
            
        # In ra lưới (lật ngược trục y để (0,0) ở góc dưới bên trái)
        print("\n".join([" ".join(row) for row in np.flipud(grid)]))
        print("-" * (self.grid_size * 2 - 1))