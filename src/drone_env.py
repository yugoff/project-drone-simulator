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

        # Инициализация PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(os.path.dirname(__file__))
        p.setGravity(0, 0, -9.81)

        # Загружаем объекты
        self.water_plane = p.loadURDF('environment/plane.urdf', basePosition=[0, 0, 0], useFixedBase=True)
        p.changeVisualShape(self.water_plane, -1, rgbaColor=[0, 0, 1, 1])
        self.platform = p.loadURDF('environment/platform.urdf', basePosition=[0, 0, 0.5], useFixedBase=True)
        self.drone = p.loadURDF('environment/drone.urdf', basePosition=[0, 0, 3.55], useFixedBase=False)

        # Пространство наблюдений: позиция дрона (3), лин. скорость (3), углы Эйлера (3), 
        # угл. скорость (3), позиция платформы (3)
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, 0, -2, -2, -2, -np.pi, -np.pi, -np.pi, -5, -5, -5, -2, -2, 0], dtype=np.float32),
            high=np.array([5, 5, 5, 2, 2, 2, np.pi, np.pi, np.pi, 5, 5, 5, 2, 2, 1], dtype=np.float32)
        )

        # Пространство действий: тяга для 4 моторов
        self.action_space = spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))

        # Позиции моторов
        self.motor_positions = [
            [0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, 0.2, 0], [-0.2, -0.2, 0]
        ]

        # Параметры движения платформы (волны)
        self.time_step = 0
        self.wave_amplitude = 0.2  # Амплитуда колебаний по высоте (м)
        self.wave_frequency = 0.5  # Частота колебаний (Гц)
        self.wave_phase = 0        # Фаза для разнообразия

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        self.water_plane = p.loadURDF('environment/plane.urdf', basePosition=[0, 0, 0], useFixedBase=True)
        p.changeVisualShape(self.water_plane, -1, rgbaColor=[0, 0, 1, 1])
        self.platform = p.loadURDF('environment/platform.urdf', basePosition=[0, 0, 0.5], useFixedBase=True)
        self.drone = p.loadURDF('environment/drone.urdf', basePosition=[0, 0, 3.55], useFixedBase=False)

        p.resetBaseVelocity(self.drone, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        
        # Сброс времени для движения платформы
        self.time_step = 0
        self.wave_phase = np.random.uniform(0, 2 * np.pi)  # Случайная начальная фаза

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

        # Обновляем позицию платформы (имитация волн)
        self.time_step += 1 / 240  # PyBullet обычно работает на 240 Гц
        platform_z = 0.5 + self.wave_amplitude * np.sin(2 * np.pi * self.wave_frequency * self.time_step + self.wave_phase)
        platform_x = 0.3 * np.sin(2 * np.pi * 0.2 * self.time_step)  # Лёгкое движение по X
        platform_y = 0.3 * np.cos(2 * np.pi * 0.2 * self.time_step)  # Лёгкое движение по Y
        p.resetBasePositionAndOrientation(self.platform, [platform_x, platform_y, platform_z], [0, 0, 0, 1])

        # Шаг симуляции
        p.stepSimulation()

        # Наблюдения
        observation = self._get_observation()
        pos = observation[:3]          # Позиция дрона
        lin_vel = observation[3:6]     # Линейная скорость
        euler_angles = observation[6:9] # Углы Эйлера
        ang_vel = observation[9:12]    # Угловая скорость
        platform_pos = observation[12:15]  # Позиция платформы

        lin_vel = np.clip(lin_vel, -5, 5)
        ang_vel = np.clip(ang_vel, -5, 5)

        # Расчёт расстояния до платформы (динамическая цель)
        distance = np.linalg.norm(pos - platform_pos)

        # Штраф за скорость
        speed_penalty = np.linalg.norm(lin_vel) * 0.1 + np.linalg.norm(ang_vel) * 0.05

        # Штраф за ориентацию
        roll, pitch, yaw = euler_angles
        angle_penalty = 0
        if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
            angle_penalty = 10

        # Награда: близость к платформе и стабильность
        height_error = abs(pos[2] - platform_pos[2])
        reward = 10 - distance - 0.5 * np.sum(np.abs(euler_angles)) - speed_penalty - angle_penalty - 2 * height_error
        # Бонус за посадку
        if distance < 0.1 and height_error < 0.05:
            reward += 50
            terminated = True
        else:
            terminated = False

        reward = float(max(reward, -20))  # Преобразуем в float и ограничиваем

        # Условия завершения
        terminated = terminated or bool(pos[2] < 0.2 or abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2)
        truncated = bool(pos[2] > 7)

        info = {'distance': float(distance), 'height': float(pos[2]), 'platform_height': float(platform_pos[2])}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        time.sleep(0.01)

    def close(self):
        p.disconnect(self.physics_client)

    def _get_observation(self):
        # Позиция и ориентация дрона
        pos, ori_quat = p.getBasePositionAndOrientation(self.drone)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone)
        euler_angles = p.getEulerFromQuaternion(ori_quat)

        # Позиция платформы
        platform_pos, _ = p.getBasePositionAndOrientation(self.platform)

        # Объединяем: позиция дрона, скорости, углы, позиция платформы
        return np.concatenate([pos, lin_vel, euler_angles, ang_vel, platform_pos], dtype=np.float32)

if __name__ == '__main__':
    env = DroneEnv()
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f'Drone Height: {info['height']:.2f}, Platform Height: {info['platform_height']:.2f}, Reward: {reward:.2f}')
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()