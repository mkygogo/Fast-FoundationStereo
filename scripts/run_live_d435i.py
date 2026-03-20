# scripts/run_live_d435i.py
import os, sys, cv2, torch, logging, yaml, argparse, time
import numpy as np
import pyrealsense2 as rs

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 全局变量：用于记录鼠标在窗口中的实时位置
# ==========================================
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth', type=str)
    parser.add_argument('--valid_iters', type=int, default=6, help='推理迭代次数，越低越快')
    parser.add_argument('--max_disp', type=int, default=192, help='最大视差')
    parser.add_argument('--scale', default=1.0, type=float, help='图像缩放比例')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # 1. 加载模型
    logging.info("正在加载 Fast-FoundationStereo 模型...")
    with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
        cfg: dict = yaml.safe_load(ff)
    for k in args.__dict__:
        if args.__dict__[k] is not None:
            cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)

    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()
    logging.info("模型加载完成！")

    # 2. 准备 OpenCV 交互窗口
    window_name = 'Live Depth (Left: IR, Right: Fast-FS)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 3. 初始化 D435i 管道配置
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 424, 240, rs.format.y8, 15)
    config.enable_stream(rs.stream.infrared, 2, 424, 240, rs.format.y8, 15)

    pipeline_started = False
    cam_fx = 0.0
    cam_baseline = 0.0
    
    logging.info("开始实时检测... 移动鼠标查看距离，按 ESC 退出。")

    try:
        while True:
            # --- 深度自愈状态机与硬件参数提取 ---
            if not pipeline_started:
                try:
                    ctx = rs.context()
                    if len(ctx.devices) == 0:
                        logging.warning("物理设备丢失，等待系统重新挂载 USB...")
                        time.sleep(2)
                        continue
                    
                    logging.info("发现相机，正在执行底层硬件复位...")
                    ctx.devices[0].hardware_reset()
                    time.sleep(3) 

                    profile = pipeline.start(config)
                    device = profile.get_device()
                    depth_sensor = device.query_sensors()[0]
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        depth_sensor.set_option(rs.option.emitter_enabled, 0)
                    
                    # 动态读取相机的内参 fx 和物理基线长度
                    ir1_profile = profile.get_stream(rs.stream.infrared, 1)
                    ir2_profile = profile.get_stream(rs.stream.infrared, 2)
                    cam_fx = ir1_profile.as_video_stream_profile().get_intrinsics().fx
                    cam_baseline = abs(ir2_profile.get_extrinsics_to(ir1_profile).translation[0])
                    
                    pipeline_started = True
                    logging.info("👉 相机已连接！")
                    for _ in range(3):
                        pipeline.wait_for_frames(5000)
                except Exception as e:
                    logging.error(f"恢复失败，稍后重试: {e}")
                    time.sleep(2)
                    continue

            # --- 正常推理流程 ---
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError as e:
                pipeline_started = False
                try: pipeline.stop()
                except: pass
                continue 

            ir1_frame = frames.get_infrared_frame(1)
            ir2_frame = frames.get_infrared_frame(2)
            if not ir1_frame or not ir2_frame: continue

            ir1 = np.asanyarray(ir1_frame.get_data())
            ir2 = np.asanyarray(ir2_frame.get_data())
            img0 = np.tile(ir1[..., None], (1, 1, 3))
            img1 = np.tile(ir2[..., None], (1, 1, 3))

            H, W = img0.shape[:2]

            img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            
            padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
            img0_pad, img1_pad = padder.pad(img0_tensor, img1_tensor)

            # PyTorch 模型推理
            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = model.forward(img0_pad, img1_pad, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')

            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

            # --- 核心：将视差转换为物理深度矩阵 (单位：米) ---
            # 引入 1e-5 防止除以 0 的非法计算
            depth_map = (cam_fx * args.scale * cam_baseline) / (disp + 1e-5)

            # 可视化处理与修复 RGB/BGR 错位 Bug
            vis = vis_disparity(disp, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            img0_bgr = cv2.cvtColor(img0.astype(np.uint8), cv2.COLOR_RGB2BGR)
            combined_view = np.concatenate([img0_bgr, vis_bgr], axis=1)

            # 将画面放大 2 倍以方便观看
            display_view = cv2.resize(combined_view, (combined_view.shape[1]*2, combined_view.shape[0]*2))
            disp_H, disp_W = display_view.shape[:2]

            # --- 鼠标交互与深度信息绘制 ---
            # 如果鼠标尚未移入窗口，默认探测左图的正中心点
            if mouse_x < 0 or mouse_y < 0:
                probe_x, probe_y = disp_W // 4, disp_H // 2
            else:
                probe_x, probe_y = mouse_x, mouse_y
                
            # 将鼠标在放大 2 倍的窗口上的坐标，映射回原始 HxW 分辨率
            orig_x = probe_x // 2
            orig_y = probe_y // 2
            
            # 判断鼠标是在左半屏(红外图)还是右半屏(深度图)，并统一换算为单张图的像素 X 坐标
            pixel_x = orig_x - W if orig_x >= W else orig_x
            pixel_y = orig_y
            
            # 防止索引越界
            if 0 <= pixel_x < W and 0 <= pixel_y < H:
                # 从深度矩阵中读取该像素的真实距离
                d_val = depth_map[pixel_y, pixel_x]
                
                # 在左侧红外图和右侧深度图上，同步画出绿色的十字瞄准星
                left_center = (pixel_x * 2, pixel_y * 2)
                right_center = ((pixel_x + W) * 2, pixel_y * 2)
                cv2.drawMarker(display_view, left_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.drawMarker(display_view, right_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                # 绘制距离文字 (带黑色描边以防止在强光/全白背景下看不清)
                text = f"Dist: {d_val:.3f} m"
                cv2.putText(display_view, text, (probe_x + 15, probe_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(display_view, text, (probe_x + 15, probe_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            cv2.imshow(window_name, display_view)
            
            if cv2.waitKey(1) == 27:
                break

    finally:
        try:
            if pipeline_started: pipeline.stop()
        except: pass
        cv2.destroyAllWindows()
        logging.info("已安全退出。")