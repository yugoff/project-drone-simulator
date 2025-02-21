import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import numpy as np
import time
import pybullet_data
import os

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Инициализация клиента PyBullet с графическим интерфейсом
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Для plane.urdf
        p.setAdditionalSearchPath(os.path.dirname(__file__))    # Для пользовательских URDF
        p.setGravity(0, 0, -9.81)

        # Загружаем объекты
        self.water_plane = p.loadURDF('environment/plane.urdf', basePosition=[0, 0, 0], useFixedBase=True)
        p.changeVisualShape(self.water_plane, -1, rgbaColor=[0, 0, 1, 1])
        self.platform = p.loadURDF('environment/platform.urdf', basePosition=[0, 0, 0.5], useFixedBase=True)
        self.drone = p.loadURDF('environment/drone.urdf', basePosition=[0, 0, 3.55], useFixedBase=False)

        # Пространство наблюдений: позиция, лин. скорость, углы Эйлера, угл. скорость
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, 0, -2, -2, -2, -np.pi, -np.pi, -np.pi, -5, -5, -5], dtype=np.float32),
            high=np.array([5, 5, 5, 2, 2, 2, np.pi, np.pi, np.pi, 5, 5, 5], dtype=np.float32)
        )

        # Пространство действий: тяга для 4 моторов
        self.action_space = spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))

        # Позиции моторов
        self.motor_positions = [
            [0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, 0.2, 0], [-0.2, -0.2, 0]
        ]

        # Целевая позиция: чуть выше платформы
        self.target_pos = np.array([0, 0, 0.55], dtype=np.float32)

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        self.water_plane = p.loadURDF('environment/plane.urdf', basePosition=[0, 0, 0], useFixedBase=True)
        p.changeVisualShape(self.water_plane, -1, rgbaColor=[0, 0, 1, 1])
        self.platform = p.loadURDF('environment/platform.urdf', basePosition=[0, 0, 0.5], useFixedBase=True)
        self.drone = p.loadURDF('environment/drone.urdf', basePosition=[0, 0, 3.55], useFixedBase=False)

        p.resetBaseVelocity(self.drone, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        return self._get_observation(), {}

    def step(self, action):
        # Ограничиваем действия
        action = np.clip(action, -1, 1)

        # Преобразуем в тягу: [-1, 1] -> [0, 10] Н
        thrusts = (action + 1) / 2 * 10

        # Применяем силы к моторам
        for i, thrust in enumerate(thrusts):
            p.applyExternalForce(
                self.drone, -1, [0, 0, thrust], self.motor_positions[i], p.WORLD_FRAME
            )

        # Шаг симуляции
        p.stepSimulation()

        # Наблюдения
        observation = self._get_observation()
        pos = observation[:3]
        lin_vel = observation[3:6]
        euler_angles = observation[6:9]
        ang_vel = observation[9:12]

        lin_vel = np.clip(lin_vel, -5, 5)
        ang_vel = np.clip(ang_vel, -5, 5)

        # Расчёт расстояния до цели
        distance = np.linalg.norm(pos - self.target_pos)

        # Штраф за скорость
        speed_penalty = np.linalg.norm(lin_vel) * 0.1 + np.linalg.norm(ang_vel) * 0.05

        # Штраф за ориентацию
        roll, pitch, yaw = euler_angles
        angle_penalty = 0
        if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
            angle_penalty = 10

        # Награда: положительная за близость к платформе, штрафы за отклонения
        height_error = abs(pos[2] - self.target_pos[2])  # Ошибка по высоте
        reward = 10 - distance - 0.5 * np.sum(np.abs(euler_angles)) - speed_penalty - angle_penalty - 2 * height_error
        reward = float(max(reward, -20))  # Преобразуем в обычный float и ограничиваем снизу

        # Условия завершения
        terminated = bool(pos[2] < 0.2 or abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2)  # Падение или переворот
        truncated = bool(pos[2] > 7)  # Улёт выше 7 м

        # Дополнительная информация
        info = {'distance': float(distance), 'height': float(pos[2])}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        time.sleep(0.01)

    def close(self):
        p.disconnect(self.physics_client)

    def _get_observation(self):
        pos, ori_quat = p.getBasePositionAndOrientation(self.drone)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone)
        euler_angles = p.getEulerFromQuaternion(ori_quat)
        return np.concatenate([pos, lin_vel, euler_angles, ang_vel], dtype=np.float32)

if __name__ == '__main__':
    env = DroneEnv()
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Случайное действие
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f'Height: {info['height']:.2f}, Reward: {reward:.2f}, Distance: {info['distance']:.2f}')
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()