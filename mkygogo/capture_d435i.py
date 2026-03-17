import pyrealsense2 as rs
import numpy as np
import cv2
import os

def capture_stereo_data(output_dir="d435i_data"):
    # 创建保存数据的目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ctx = rs.context()
    if len(ctx.devices) > 0:
        print("检测到 D435i，正在执行硬件重置以清理残留状态...")
        ctx.devices[0].hardware_reset()
        import time
        time.sleep(3) # 给相机 3 秒钟的重启和重新连接时间
    else:
        print("未检测到 RealSense 设备，请检查 USB 连接！")
        return

    # 1. 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # D435i 的双目测距使用的是左右两个红外相机 (Infrared 1 和 2)
    # 我们配置获取 640x480 分辨率的红外图像
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)

    print("正在启动 D435i 相机...")
    profile = pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0) # 0为关闭，1为开启
        print("已强制关闭红外发射器以降低 USB 功耗。")

    try:
        # 2. 获取相机的内参和基线
        ir1_profile = profile.get_stream(rs.stream.infrared, 1)
        # ... (获取内参保存 K.txt 的代码保持不变，确保这部分还在) ...

        # 3. 光速抓图（放弃 10 帧预热，改为只抓取第 2 帧）
        print("正在抓取图像...")
        
        # 抓取第 1 帧 (通常刚启动的第 1 帧曝光不稳定或有残影，我们丢弃它)
        pipeline.wait_for_frames(10000) 
        
        # 抓取第 2 帧 (作为我们的目标帧)
        frames = pipeline.wait_for_frames(10000)
        
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)

        if not ir1_frame or not ir2_frame:
            print("无法获取红外帧！")
            return

        # 将帧转换为 numpy 数组以便保存
        image_left = np.asanyarray(ir1_frame.get_data())
        image_right = np.asanyarray(ir2_frame.get_data())

        # 保存图片
        left_path = os.path.join(output_dir, "d435i_left.png")
        right_path = os.path.join(output_dir, "d435i_right.png")
        cv2.imwrite(left_path, image_left)
        cv2.imwrite(right_path, image_right)
        print(f"🎉 成功！已保存左图至: {left_path}")
        print(f"🎉 成功！已保存右图至: {right_path}")

    finally:
        # 如果相机已经掉线，调用 stop() 会报错，这里加个 try-except 忽略掉线报错
        try:
            pipeline.stop()
        except:
            pass
        print("相机连接已释放。")

if __name__ == "__main__":
    capture_stereo_data()