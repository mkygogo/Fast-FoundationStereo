# scripts/run_live_d435i.py
import os, sys, cv2, torch, logging, yaml, argparse, time
import numpy as np
import pyrealsense2 as rs

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity

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

    # ==========================================
    # 1. 初始化模型 (保持不变)
    # ==========================================
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

    # ==========================================
    # 2. 硬件重置
    # ==========================================
    ctx = rs.context()
    if len(ctx.devices) > 0:
        logging.info("检测到 D435i，正在执行硬件重置...")
        ctx.devices[0].hardware_reset()
        time.sleep(3) # 等待相机重启
    else:
        logging.error("未检测到相机，请检查连接！")
        sys.exit(1)

    # ==========================================
    # 3. 初始化 D435i 配置
    # ==========================================
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)
    
    pipeline_started = False
    logging.info("开始实时深度检测... 按 ESC 键退出")

    # ==========================================
    # 4. 具备自愈能力的实时推理主循环
    # ==========================================
    try:
        while True:
            # --- 状态机：如果通道未启动或掉线，自动执行启动流程 ---
            if not pipeline_started:
                try:
                    profile = pipeline.start(config)
                    # 强制关闭红外发射器以降低功耗
                    device = profile.get_device()
                    depth_sensor = device.query_sensors()[0]
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        depth_sensor.set_option(rs.option.emitter_enabled, 0)
                    pipeline_started = True
                    logging.info("👉 相机通道已成功(重新)连接！")
                    
                    # 刚启动时丢弃1帧坏帧
                    pipeline.wait_for_frames(5000)
                except Exception as e:
                    logging.warning(f"相机启动失败，等待1秒后重试... ({e})")
                    time.sleep(1)
                    continue

            # --- 正常抓取与推理流程 ---
            try:
                # 抓取双目帧 (设置 5000ms 超时)
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError as e:
                # ！！！核心自愈逻辑：检测到掉线，重置状态，停止废弃通道 ！！！
                logging.error(f"❌ 数据流异常中断 ({e})，正在尝试硬件重连...")
                pipeline_started = False
                try:
                    pipeline.stop()
                except:
                    pass
                continue

            ir1_frame = frames.get_infrared_frame(1)
            ir2_frame = frames.get_infrared_frame(2)
            
            if not ir1_frame or not ir2_frame:
                continue

            # 转换为 numpy 数组并扩展为 3 通道
            ir1 = np.asanyarray(ir1_frame.get_data())
            ir2 = np.asanyarray(ir2_frame.get_data())
            img0 = np.tile(ir1[..., None], (1, 1, 3))
            img1 = np.tile(ir2[..., None], (1, 1, 3))

            H, W = img0.shape[:2]

            img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            
            padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
            img0_pad, img1_pad = padder.pad(img0_tensor, img1_tensor)

            # 运行网络推理
            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = model.forward(img0_pad, img1_pad, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')

            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

            # 视差图可视化
            vis = vis_disparity(disp, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
            
            img0_bgr = cv2.cvtColor(img0.astype(np.uint8), cv2.COLOR_RGB2BGR)
            combined_view = np.concatenate([img0_bgr, vis], axis=1)

            cv2.imshow('Live Fast-FoundationStereo Depth (Left: IR, Right: Depth)', combined_view)
            
            if cv2.waitKey(1) == 27:
                break

    finally:
        try:
            if pipeline_started:
                pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()
        logging.info("已安全退出视频流。")