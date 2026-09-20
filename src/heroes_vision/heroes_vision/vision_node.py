import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
import cv2
import numpy as np
import os

from heroes_interfaces.msg import TowerStatus
from sensor_msgs.msg import Image

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.declare_parameter('source', 0, ParameterDescriptor(dynamic_typing=True))
        param_val = self.get_parameter('source').value
        
        if isinstance(param_val, int):
            video_source = param_val
        else:
            try:
                video_source = int(param_val)
            except ValueError:
                video_source = os.path.abspath(os.path.expanduser(param_val))
                
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            self.get_logger().error(f'GAGAL MEMBUKA SOURCE: {video_source}')
            return
            
        self.status_pub = self.create_publisher(TowerStatus, '/vision/tower_status', 10)
        self.debug_pub = self.create_publisher(Image, '/vision/debug_image', 10)
        
        self.timer = self.create_timer(0.2, self.timer_callback)

    def get_color_mask(self, hsv_frame):
        # Rentang HSV yang disesuaikan untuk mengantisipasi pencahayaan ruangan redup/kekuningan
        # Mask Biru
        lower_blue = np.array([90, 100, 40])
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        
        # Mask Merah (Rentang 1 & 2)
        lower_red1 = np.array([0, 100, 40])
        upper_red1 = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        
        lower_red2 = np.array([170, 100, 40])
        upper_red2 = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        return mask_blue, mask_red

    def analyze_sblock(self, b_roi, r_roi):
        h, w = b_roi.shape
        b_count = cv2.countNonZero(b_roi)
        r_count = cv2.countNonZero(r_roi)
        
        if b_count < 300 and r_count < 300: return 0
        if b_count > r_count * 3: return 1
        if r_count > b_count * 3: return 2
        
        half_w, half_h = w // 2, h // 2
        b_top, b_bot = cv2.countNonZero(b_roi[:half_h, :]), cv2.countNonZero(b_roi[half_h:, :])
        b_lef, b_rig = cv2.countNonZero(b_roi[:, :half_w]), cv2.countNonZero(b_roi[:, half_w:])
        
        r_top, r_bot = cv2.countNonZero(r_roi[:half_h, :]), cv2.countNonZero(r_roi[half_h:, :])
        r_lef, r_rig = cv2.countNonZero(r_roi[:, :half_w]), cv2.countNonZero(r_roi[:, half_w:])
        
        if b_top > r_top and r_bot > b_bot: return 3
        if b_rig > r_rig and r_lef > b_lef: return 4
        if b_bot > r_bot and r_top > b_top: return 5
        if b_lef > r_lef and r_rig > b_rig: return 6
        
        return 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue, mask_red = self.get_color_mask(hsv)
        
        msg = TowerStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.sblock = 0
        msg.eblock2 = 0
        msg.eblock1 = 0

        mask_total = cv2.bitwise_or(mask_blue, mask_red)
        
        # Operasi morfologi untuk membersihkan noise kecil pada lantai/background
        kernel = np.ones((5,5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Threshold area disesuaikan agar objek balok di video tertangkap dengan baik
            if area > 800: 
                x, y, w, h = cv2.boundingRect(cnt)
                valid_blocks.append({
                    'roi_b': mask_blue[y:y+h, x:x+w],
                    'roi_r': mask_red[y:y+h, x:x+w],
                    'y': y,
                    'rect': (x, y, w, h)
                })

        # Urutkan berdasarkan posisi Y dari atas ke bawah layar
        valid_blocks = sorted(valid_blocks, key=lambda b: b['y'])

        # Mapping tumpukan balok (Sesuai struktur video tumpukan 3 tingkat)
        if len(valid_blocks) >= 1:
            b0 = valid_blocks[0] # Balok paling atas (Sky Block / eblock tergantung skenario)
            msg.sblock = self.analyze_sblock(b0['roi_b'], b0['roi_r'])
            self.draw_debug(frame, b0['rect'], f"sblock: {msg.sblock}")

        if len(valid_blocks) >= 2:
            b1 = valid_blocks[1] # Balok tengah
            if cv2.countNonZero(b1['roi_b']) > cv2.countNonZero(b1['roi_r']): msg.eblock2 = 1
            else: msg.eblock2 = 2
            self.draw_debug(frame, b1['rect'], f"eblock2: {msg.eblock2}")

        if len(valid_blocks) >= 3:
            b2 = valid_blocks[2] # Balok bawah
            if cv2.countNonZero(b2['roi_b']) > cv2.countNonZero(b2['roi_r']): msg.eblock1 = 1
            else: msg.eblock1 = 2
            self.draw_debug(frame, b2['rect'], f"eblock1: {msg.eblock1}")

        self.status_pub.publish(msg)
        
        # Rakit pesan gambar debug secara manual (bypass cv_bridge)
        debug_msg = Image()
        debug_msg.header.stamp = msg.header.stamp
        debug_msg.header.frame_id = 'camera_link'
        debug_msg.height = frame.shape[0]
        debug_msg.width = frame.shape[1]
        debug_msg.encoding = 'bgr8'
        debug_msg.is_bigendian = 0
        debug_msg.step = frame.shape[1] * 3
        debug_msg.data = frame.tobytes()
        
        self.debug_pub.publish(debug_msg)

    def draw_debug(self, frame, rect, label):
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()