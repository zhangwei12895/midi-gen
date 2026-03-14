# 🎵 MIDI/GEN — AI 8-bit 音乐生成器

> 基于 Transformer 的端到端 MIDI 音乐生成系统，支持多曲风、完整歌曲结构，CPU 推理部署，配备实时 Web 播放界面。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-Web%20API-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ 项目简介

MIDI/GEN 是一个从零训练的 AI 音乐生成项目。模型学习了完整的歌曲结构（前奏→主歌→副歌→桥段→尾奏），能够根据调性、速度、曲风参数自主生成包含旋律、贝斯、和声三轨的完整 MIDI 作品，并通过 Web 界面实时试听。

**核心特点：**
- 自定义 Transformer 架构，词汇表 305 tokens，专为 MIDI 序列设计
- 完整歌曲结构感知（不是随机片段，是有头有尾的完整作品）
- 纯 CPU 推理，无需 GPU，可部署于普通云服务器
- 内置 Web 播放器，Tone.js + SoundFont 实时渲染音频
- 收藏夹、历史记录、多模型切换等完整功能

---

## 📁 项目结构

```
midi-gen/
├── app.py                  # Flask Web 服务器
├── generate.py             # 推理引擎（歌曲生成）
├── config.py               # 全局配置（模型/生成参数）
├── train.py                # 训练入口
│
├── data/
│   ├── tokenizer.py        # MIDI → token 序列
│   ├── midi_processor.py   # token → MIDI 文件
│   └── dataset.py          # 训练数据集加载
│
├── model/
│   ├── transformer.py      # Transformer 模型定义
│   └── music_theory.py     # 乐理约束（调性/和弦）
│
├── utils/
│   └── midi_utils.py       # MIDI 工具函数
│
├── checkpoints/            # 模型权重（.pt 文件）
├── outputs/                # 生成的 MIDI 文件（最多 50 个）
├── web/
│   └── index.html          # 前端界面
│
└── logs/
    └── server.log          # 服务器日志
```

---

## 🏗️ 模型架构

| 参数 | 值 |
|---|---|
| 架构 | Decoder-only Transformer |
| 词汇表大小 | 305 tokens |
| 轨道 | 旋律 (Melody) / 贝斯 (Bass) / 和声 (Harmony) |
| 推理设备 | CPU（多线程并行） |

**Token 结构：**
```
[BOS] KEY TEMPO [INTENSITY] [PITCH_CENTER]
  [SECTION] [BAR] [DENSITY] [CHORD]
    [TRACK] BEAT NOTE_ON NOTE_DUR VEL ...
[EOS]
```

**曲风（Style）对应生成结构：**

| Style | 名称 | 段落结构 |
|---|---|---|
| 0 | 舒缓 calm | INTRO → OUTRO |
| 1 | 轻快 light | INTRO → VERSE → CHORUS → OUTRO |
| 2 | 均衡 medium | INTRO → VERSE → CHORUS → VERSE → CHORUS → OUTRO |
| 3 | 充沛 energetic | + BRIDGE |
| 4 | 激情 intense | + 双 CHORUS 结尾 |

---

## 🚀 快速开始

### 环境要求

```
Python 3.10+
torch (CPU 版本)
pretty_midi
flask flask-cors
```

### 安装依赖

```bash
# 普通环境
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pretty_midi flask flask-cors

# Ubuntu 系统 Python（如宝塔面板）
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip install pretty_midi --break-system-packages
pip install flask flask-cors --break-system-packages --ignore-installed
```

### 训练模型

```bash
python train.py
```

训练完成后权重保存至 `checkpoints/best_model.pt`。

### 命令行生成

```bash
# 生成一首均衡风格的歌曲（C大调，120BPM）
python generate.py --style 2 --key C_maj --tempo 120

# 生成激情风格，温度0.9
python generate.py --style 4 --key A_min --tempo 150 --temperature 0.9

# 限制最多32小节
python generate.py --style 2 --max_bars 32
```

---

## 🌐 Web 服务部署

### 本地运行

```bash
python app.py --host 127.0.0.1 --port 5000
```

访问 `http://localhost:5000`

### 服务器部署（推荐用 screen）

```bash
# 新建 screen 会话
screen -S midi_gen

# 在 screen 内启动服务
cd /www/wwwroot/8bit_music_gen
python3 app.py --host 0.0.0.0 --port 5000 > logs/server.log 2>&1 &

# 挂起会话（保持后台运行）
Ctrl+A  然后  D
```

**screen 常用管理命令：**

```bash
# 查看所有会话
screen -ls

# 进入已有会话（假设窗口号 2269）
screen -r 2269

# 外部直接停止服务
screen -S 2269 -X stuff "pkill -f app.py\n"

# 外部直接重启服务
screen -S 2269 -X stuff "cd /www/wwwroot/8bit_music_gen && python3 app.py --host 0.0.0.0 --port 5000 > logs/server.log 2>&1 &\n"

# 查看日志
tail -f /www/wwwroot/8bit_music_gen/logs/server.log
```

### Nginx 反向代理配置

```nginx
location / {
    proxy_pass         http://127.0.0.1:5000;
    proxy_set_header   Host $host;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
}
```

---

## 🔌 API 文档

| 端点 | 方法 | 说明 |
|---|---|---|
| `GET /` | GET | 主页面 |
| `GET /api/status` | GET | 模型状态、可用模型列表 |
| `POST /api/generate` | POST | 生成 MIDI |
| `GET /api/presets` | GET | 历史文件列表（含时长） |
| `GET /api/models` | GET | 扫描 checkpoints/ 返回模型列表 |
| `POST /api/heartbeat` | POST | 前端心跳（防超时取消） |
| `GET /outputs/<filename>` | GET | 下载 MIDI 文件 |

**POST /api/generate 请求体：**

```json
{
  "style":       2,        // 曲风 0-4
  "key":         "C_maj",  // 调性
  "tempo":       120,      // BPM
  "temperature": 0.9,      // 采样温度
  "top_k":       40,
  "top_p":       0.9,
  "max_bars":    8,        // 最多小节数（0=不限）
  "model":       "best_model.pt"  // 可选，指定模型文件名
}
```

**响应：**

```json
{
  "url":      "/outputs/gen_20250312_143022_s2_medium_t090.mid",
  "filename": "gen_20250312_143022_s2_medium_t090.mid",
  "duration": 42.5,
  "bars":     16,
  "notes":    380,
  "sections": ["INTRO", "VERSE", "CHORUS", "OUTRO"],
  "elapsed":  18,
  "style":    2,
  "key":      "C_maj",
  "tempo":    120
}
```

---

## 🖥️ Web 界面功能

- **多曲风选择**：5档强度，实时预览段落结构
- **参数控制**：调性、速度、随机度、最大小节数
- **多模型切换**：自动扫描 checkpoints/，下拉切换
- **实时播放**：Tone.js + SoundFont，无需额外插件
- **音量控制**：GainNode 实时生效，拖动立即响应
- **收藏功能**：localStorage 持久化，收藏列表置顶显示
- **历史记录**：最多展示 30 条，支持点击直接播放
- **下载**：直接下载 MIDI 文件到本地
- **心跳保活**：用户离开 3 分钟自动停止生成，节省服务器资源

---

## 📋 注意事项

- `checkpoints/` 目录下的 `.pt` 文件**不建议上传到 GitHub**（文件通常较大），可使用 [Git LFS](https://git-lfs.github.com/) 或单独存储
- `outputs/` 目录下生成的 MIDI 文件也建议加入 `.gitignore`
- 旧版本 checkpoint（v3 及以前架构）与 v4 不兼容，需重新训练

---

## 📄 License

MIT License

---

## 🙏 致谢

- [Tone.js](https://tonejs.github.io/) — Web Audio 音序引擎
- [soundfont-player](https://github.com/danigb/soundfont-player) — SoundFont 渲染
- [pretty_midi](https://github.com/craffel/pretty-midi) — MIDI 处理
- [@tonejs/midi](https://github.com/Tonejs/Midi) — MIDI 解析