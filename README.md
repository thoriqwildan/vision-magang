# HEROES Gadjah Mada Robotic Team — Vision Node ROS 2

Node ROS 2 berbasis Python (`vision_node`) untuk mendeteksi status *Building Spot* (tumpukan *Earth Block* dan *Sky Block*) menggunakan OpenCV, serta mempublikasikannya ke *topic* `/vision/tower_status`.

---

## Prasyarat Sistem

* **OS:** Ubuntu 22.04 LTS
* **ROS 2:** Humble
* **Python:** 3.10+

---

## 1. Buka terminal
```
git clone https://github.com/thoriqwildan/vision-magang.git

cd vision-magang

pip install "numpy<2>"

colcon build --packages-select heroes_interfaces heroes_vision

source install/setup.bash

ros2 run heroes_vision vision_node --ros-args -p source:="src/heroes_vision/sample-videos/1.mp4"

```

## 2. Buka terminal kedua
```
source ~/heroes_ws/install/setup.bash

ros2 topic echo /vision/tower_status
```

## 3. Buka terminal ketiga
```
ros2 run rqt_image_view rqt_image_view
```

## 4. Pilih topic /vision/debug_image
