import pybullet
import pybullet_data
import cv2
import math
import numpy as np

# Подключение к PyBullet (GUI)
pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setGravity(0, 0, -9.81)

# Загрузка окружающей среды, платформы и дрона
environment_id = pybullet.loadURDF('environment.urdf')
platform_id = pybullet.loadURDF('platform.urdf', basePosition=[0, 0, 1])
drone_id = pybullet.loadURDF('drone.urdf', basePosition=[0, 0, 2])

# Инициализация камеры дрона
camera_params = {
    "width": 640, "height": 480,  # Разрешение камеры
    "view_matrix": pybullet.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], 
                                                              distance=5, yaw=90, pitch=-30, roll=0, 
                                                              upAxisIndex=2),  # Положение и ориентация камеры, ось Z вверх
    "projection_matrix": pybullet.computeProjectionMatrixFOV(fov=60, aspect=640/480, nearVal=0.1, farVal=100)
}

# Функция для симуляции волн на платформе (синусоидальные колебания)
def apply_wave_motion():
    current_time = pybullet.getRealTimeSimulation()  # Время реального времени симуляции
    displacement = 0.2 * math.sin(current_time)  # Платформа колеблется вверх-вниз
    pybullet.resetBasePositionAndOrientation(platform_id, [0, 0, 1 + displacement], [0, 0, 0, 1])

# Функция для получения изображения с камеры
def get_camera_image():
    width, height, rgb_img, depth_img, segmentation_img = pybullet.getCameraImage(
        camera_params["width"], camera_params["height"], viewMatrix=camera_params["view_matrix"], 
        projectionMatrix=camera_params["projection_matrix"]
    )
    # Преобразование изображения в OpenCV формат
    rgb_img = np.reshape(rgb_img, (height, width, 4))  # RGB изображение с альфа-каналом
    return rgb_img

# Основной цикл симуляции
for i in range(1000):
    # Симуляция колебаний платформы
    apply_wave_motion()

    # Получение изображения с камеры дрона
    image = get_camera_image()

    # Показываем изображение с камеры для проверки
    cv2.imshow("Drone Camera View", image)
    cv2.waitKey(1)  # Задержка на 1ms, чтобы обновить изображение

    # Обновление симуляции
    pybullet.stepSimulation()

# Закрыть окно и завершить симуляцию
cv2.destroyAllWindows()
pybullet.disconnect()