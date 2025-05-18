import pygame
import numpy as np
import time
import json
import os
import random
import datetime

####################################################
# This script generates stimuli for the experiment.
# Author: Gui Yun
# Email: guiy24@mails.tsinghua.edu.cn
# Date: 2025-05-18
# Version: 2.1
# Description: This script generates orientation gratings stimuli for the experiment (pygame version, config support).

# Experiment Name: Visual Orientation Discrimination
# TODO 1: When use this script, please fill in the following information:
# Student Name: 
# Student ID:
# Date:
# Experiment Date:
# Experiment Group:
# Mice ID:
####################################################



# ========== 默认参数定义 ==========
def get_default_config():
    return {
        "stim_num": 20,  # 每种朝向的刺激个数
        "stim_duration": 4.0,
        "isi": 6.0,
        "stim_size": 600,  # 默认更大
        "stim_freq": 2.0,
        "stim_contrast": 1.0,
        "stim_phase": 0.0,
        "stim_eccentricity": 2.0,
        "orientations": [0, 45, 90, 135, 180, 225, 270, 315],
        "bg_color": [128, 128, 128],
        "drift_speed": 0.0,  # 像素/秒
        "play_mode": "sequential",  # sequential or random
        "fullscreen": False
    }

# ========== 读取/生成配置文件 ==========
config_path = 'config.json'
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
else:
    config = get_default_config()
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"未检测到config.json，已生成默认配置文件。请根据需要修改后重新运行。")

# ========== TODO 2: 修改下方参数以适应你的实验需求 ==========
# 注意 这里只是注释，真正修改请在config.json中修改
# stim_num：每种朝向的刺激个数
stim_num = config.get("stim_num", 20)  # int
# stim_duration：每个刺激呈现时长（秒）
stim_duration = config.get("stim_duration", 4.0)  # float, 单位：秒
# isi：刺激间隔时长（秒）
isi = config.get("isi", 6.0)  # float, 单位：秒
# stim_size：刺激图像的像素大小（建议大于屏幕高度一半）
stim_size = config.get("stim_size", 600)  # int, 单位：像素
# stim_freq：空间频率，单位cycles/degree（可理解为条纹密度）
stim_freq = config.get("stim_freq", 2.0)  # float
# stim_contrast：对比度，0~1
stim_contrast = config.get("stim_contrast", 1.0)  # float, 0~1
# stim_phase：相位，单位弧度
stim_phase = config.get("stim_phase", 0.0)  # float, 单位：弧度
# stim_eccentricity：离中心度，单位degree（暂未用到）
stim_eccentricity = config.get("stim_eccentricity", 2.0)  # float
# orientations：朝向列表，单位度
orientations = config.get("orientations", [0, 45, 90, 135, 180, 225, 270, 315])  # list[int]
# bg_color：背景色，RGB格式，0~255
bg_color = tuple(config.get("bg_color", [128, 128, 128]))  # tuple[int, int, int]
# drift_speed：光栅漂移速度，像素/秒，0为静止
drift_speed = config.get("drift_speed", 0.0)  # float
# play_mode："sequential"顺序，"random"随机
play_mode = config.get("play_mode", "random")  # str
# fullscreen：是否全屏，true为全屏，false为窗口
fullscreen = config.get("fullscreen", False)  # bool

# ========== 生成刺激序列 ==========
stimuli_sequence = []
for ori in orientations:
    for _ in range(stim_num):
        stimuli_sequence.append(ori)
if play_mode == "random":
    random.shuffle(stimuli_sequence)

# ========== 写入刺激参数到文件 ==========
exp_start_time = time.time()
exp_start_str = datetime.datetime.fromtimestamp(exp_start_time).strftime('%Y-%m-%d %H:%M:%S')
file_time_str = datetime.datetime.fromtimestamp(exp_start_time).strftime('%Y%m%d_%H%M')
stimuli_filename = f"Stimuli/stimuli_{file_time_str}.txt"

with open(stimuli_filename, 'w', encoding='utf-8') as f:
    f.write(f"实验实际开始时间: {exp_start_str}\n")
    f.write("刺激参数：\n")
    f.write(f"刺激大小: {stim_size} 像素\n")
    f.write(f"刺激频率: {stim_freq} cycles/degree\n")
    f.write(f"刺激对比度: {stim_contrast}\n")
    f.write(f"刺激相位: {stim_phase} 弧度\n")
    f.write(f"刺激持续时间: {stim_duration} 秒\n")
    f.write(f"刺激间隔: {isi} 秒\n")
    f.write(f"刺激朝向: {orientations} 度\n")
    f.write(f"刺激离中心度: {stim_eccentricity} degree\n")
    f.write(f"背景色: {bg_color}\n")
    f.write(f"光栅漂移速度: {drift_speed} 像素/秒\n")
    f.write(f"播放模式: {play_mode}\n")
    f.write(f"全屏模式: {fullscreen}\n")
    f.write("\n刺激呈现顺序（含时间）：\n")
    # 计算每个刺激的绝对呈现时间
    stim_times = []
    t = 2  # 初始灰色背景2秒
    for idx, ori in enumerate(stimuli_sequence):
        stim_times.append(t)
        t += stim_duration
        if idx < len(stimuli_sequence) - 1:
            t += isi
    for idx, (ori, stim_time) in enumerate(zip(stimuli_sequence, stim_times)):
        f.write(f"刺激{idx+1}: 朝向 {ori}°，呈现时间: {stim_time:.1f} 秒\n")

# ========== 生成正弦光栅函数 ==========
def generate_grating(size, orientation, sf, contrast, phase, drift_px=0):
    radians = np.deg2rad(orientation)
    x = np.linspace(-np.pi, np.pi, size)
    y = np.linspace(-np.pi, np.pi, size)
    xv, yv = np.meshgrid(x, y)
    # 旋转坐标
    xt = xv * np.cos(radians) + yv * np.sin(radians)
    # 漂移相位
    grating = np.sin(sf * xt + phase + drift_px)
    grating = ((grating * contrast + 1) / 2 * 255).astype(np.uint8)
    grating_rgb = np.stack([grating]*3, axis=-1)
    return grating_rgb

# ========== pygame 初始化 ==========
pygame.init()
if fullscreen:
    win_info = pygame.display.Info()
    win_size = (win_info.current_w, win_info.current_h)
    screen = pygame.display.set_mode(win_size, pygame.FULLSCREEN)
else:
    win_size = (800, 600)
    screen = pygame.display.set_mode(win_size)
pygame.display.set_caption("Orientation Grating Stimuli")

# 初始灰色背景
screen.fill(bg_color)
pygame.display.flip()
print(f"初始灰色背景 {2} 秒")
time.sleep(2)

# ========== 刺激呈现主循环 ==========
for idx, ori in enumerate(stimuli_sequence):
    t0 = time.time()
    drift_px = 0
    # 刺激呈现时长内动态漂移
    while time.time() - t0 < stim_duration:
        # 计算当前漂移相位
        if drift_speed != 0:
            drift_px = drift_speed * (time.time() - t0)
        grating_img = generate_grating(stim_size, ori, stim_freq, stim_contrast, stim_phase, drift_px)
        surf = pygame.surfarray.make_surface(np.transpose(grating_img, (1, 0, 2)))
        rect = surf.get_rect(center=(win_size[0]//2, win_size[1]//2))
        screen.fill(bg_color)
        screen.blit(surf, rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        time.sleep(0.01)
    print(f"呈现第{idx+1}个刺激，朝向: {ori}°，持续{stim_duration}秒")
    # 刺激间隔
    if idx < len(stimuli_sequence) - 1:
        screen.fill(bg_color)
        pygame.display.flip()
        print(f"刺激间隔{isi}秒\n")
        t1 = time.time()
        while time.time() - t1 < isi:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            time.sleep(0.01)

# 结束，关闭窗口
pygame.quit()
print("实验结束，窗口已关闭。")

