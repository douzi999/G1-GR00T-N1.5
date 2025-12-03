import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import threading,os
os.environ["QT_QPA_PLATFORM"] = "xcb"

class ImageClient:
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None, wrist_img_shape = None, wrist_img_shm_name = None, 
                       image_show = False, server_address = "192.168.123.164", port = 5555, Unit_Test = False):
        """
        tv_img_shape: User's expected head camera resolution shape (H, W, C). It should match the output of the image service terminal.

        tv_img_shm_name: Shared memory is used to easily transfer images across processes to the Vuer.

        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C). It should maintain the same shape as tv_img_shape.

        wrist_img_shm_name: Shared memory is used to easily transfer images.
        
        image_show: Whether to display received images in real time.

        server_address: The ip address to execute the image server script.

        port: The port number to bind to. It should be the same as the image server.

        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            try:
                self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
                self.tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = self.tv_image_shm.buf)
                self.tv_enable_shm = True
                print(f"ImageClient: 成功连接到共享内存 {tv_img_shm_name}, 尺寸: {tv_img_shape}")
            except FileNotFoundError:
                print(f"ImageClient: 无法找到共享内存 {tv_img_shm_name}")
                self.tv_enable_shm = False
        
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

        # 添加计数器控制输出频率
        self._frame_count = 0
        self._last_log_time = time.time()

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%")
    
    def _close(self):
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

    
    def receive_process(self):
        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        try:
            self._socket.connect(f"tcp://{self._server_address}:{self._port}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            print(f"\nImage client has started, connecting to {self._server_address}:{self._port}")
        except Exception as e:
            print(f"Image client connection failed: {e}")
            return

        try:
            while self.running:
                # Receive message with timeout
                try:
                    message = self._socket.recv(zmq.NOBLOCK)
                    receive_time = time.time()
                except zmq.Again:
                    # No message available, continue
                    time.sleep(0.01)
                    continue

                if self._enable_performance_eval:
                    header_size = struct.calcsize('dI')
                    try:
                        # Attempt to extract header and image data
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack('dI', header)
                    except struct.error as e:
                        print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    # No header, entire message is image data
                    jpg_bytes = message
                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                
                if current_image is None:
                    print("[Image Client] Failed to decode image.")
                    continue
                
                if self.tv_enable_shm:
                    # 修复：处理2560×480的四摄像头拼接图像
                    height, width = current_image.shape[:2]
                    
                    # 检查是否是四摄像头拼接图像 (480×2560)
                    if width == 2560 and height == 480:
                        # 根据共享内存名称决定取哪个640×480的部分
                        shm_name_lower = self.tv_image_shm.name.lower()
                        
                        if "left_high" in shm_name_lower or ("left" in shm_name_lower and "high" in shm_name_lower):
                            # 左高摄像头 - 第1部分 (0-640)
                            target_image = current_image[:, 0:640]
                        elif "right_high" in shm_name_lower or ("right" in shm_name_lower and "high" in shm_name_lower):
                            # 右高摄像头 - 第2部分 (640-1280)
                            target_image = current_image[:, 640:1280]
                        elif "left_wrist" in shm_name_lower or ("left" in shm_name_lower and "wrist" in shm_name_lower):
                            # 左手腕摄像头 - 第3部分 (1280-1920)
                            target_image = current_image[:, 1280:1920]
                        elif "right_wrist" in shm_name_lower or ("right" in shm_name_lower and "wrist" in shm_name_lower):
                            # 右手腕摄像头 - 第4部分 (1920-2560)
                            target_image = current_image[:, 1920:2560]
                        else:
                            # 默认情况，根据名称中的数字索引
                            if "0" in shm_name_lower or "first" in shm_name_lower:
                                target_image = current_image[:, 0:640]
                            elif "1" in shm_name_lower or "second" in shm_name_lower:
                                target_image = current_image[:, 640:1280]
                            elif "2" in shm_name_lower or "third" in shm_name_lower:
                                target_image = current_image[:, 1280:1920]
                            elif "3" in shm_name_lower or "fourth" in shm_name_lower:
                                target_image = current_image[:, 1920:2560]
                            else:
                                # 默认取第一个
                                target_image = current_image[:, 0:640]
                    elif width == 1280 and height == 480:
                        # 如果是1280×480的双目图像，根据共享内存名称选择正确的半部分
                        shm_name_lower = self.tv_image_shm.name.lower()
                        
                        if "left" in shm_name_lower:
                            # 左眼图像 - 取左半部分
                            target_image = current_image[:, :640]
                        elif "right" in shm_name_lower:
                            # 右眼图像 - 取右半部分
                            target_image = current_image[:, 640:]
                        else:
                            # 默认使用右半部分
                            target_image = current_image[:, 640:]
                    else:
                        # 其他尺寸的图像，直接使用
                        target_image = current_image
                    
                    # 检查尺寸匹配并写入共享内存
                    if target_image.shape == self.tv_img_array.shape:
                        np.copyto(self.tv_img_array, target_image)
                        # 减少输出频率：每100帧输出一次或每5秒输出一次
                        self._frame_count += 1
                        current_time = time.time()
                        if current_time - self._last_log_time > 5.0:
                            print(f"ImageClient: {self.tv_image_shm.name} 正常接收图像，帧率: {self._frame_count/5:.1f} fps")
                            self._frame_count = 0
                            self._last_log_time = current_time
                    else:
                        # 只在出错时输出错误信息
                        print(f"ImageClient: 尺寸不匹配！target_image: {target_image.shape}, tv_img_array: {self.tv_img_array.shape}")
                        # 尝试调整尺寸
                        try:
                            resized_image = cv2.resize(target_image, (self.tv_img_array.shape[1], self.tv_img_array.shape[0]))
                            np.copyto(self.tv_img_array, resized_image)
                            print(f"ImageClient: 调整尺寸后写入: {target_image.shape} -> {self.tv_img_array.shape}")
                        except Exception as e:
                            print(f"ImageClient: 调整尺寸失败: {e}")
                else:
                    # 只在初始化时输出一次
                    if self._frame_count == 0:
                        print("ImageClient: 共享内存未启用，无法写入数据")
                        self._frame_count = 1
                
                if self.wrist_enable_shm:
                    np.copyto(self.wrist_img_array, np.array(current_image[:, -self.wrist_img_shape[1]:]))
                
                if self._image_show:
                    height, width = 480,640
                    resized_image = cv2.resize(current_image, (width, height))
                    cv2.imshow('Image Client Stream (Right Half)', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

    
    def list_cameras(self,max_index=10):
        """探测系统中可用的摄像头索引"""
        available = []
        for index in range(max_index):
            cap = cv2.VideoCapture(index)
            if cap is None or not cap.isOpened():
                continue
            ret, _ = cap.read()
            if ret:
                available.append(index)
            cap.release()
        cv2.destroyAllWindows()  # 关闭所有窗口
        return available

if __name__ == "__main__":
    # example1
    # tv_img_shape = (480, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # img_client.receive_process()

    # example2
    # Initialize the client with performance evaluation enabled
    # client = ImageClient(image_show = True, server_address='127.0.0.1', Unit_Test=True) # local test
    client = ImageClient(image_show = True, server_address='192.168.123.164', Unit_Test=False) # deployment test
    client.receive_process()