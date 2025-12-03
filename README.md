# **GR00T微调流程**

参考文档：https://github.com/NVIDIA/Isaac-GR00T/blob/main/README.md

## **01 环境准备**

### 1、系统要求

已验证的操作系统和硬件：

- 操作系统: Ubuntu 22.04
- GPU: RTX 4090
- Python: 3.10
- CUDA: 12.4

### 2、前置依赖安装

```Python
# 安装系统依赖
sudo apt update
sudo apt install ffmpeg libsm6 libxext6 -y

# 安装 CUDA（如果尚未安装）
# 推荐 CUDA 12.4，参考: https://developer.nvidia.com/cuda-downloads
# 使用官方脚本一键安装 CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# 安装 tensorrt （可选）
# 参考：https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#
```

## **02 项目安装**

```Python
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

**注意**：CUDA  12.4 是推荐且经过官方测试的版本。不过，CUDA 11.8 也已被验证可以正常工作。如果使用 CUDA 11.8，请确保手动安装兼容版本的  flash-attn（例如，已确认 flash-attn==2.8.2 可在 CUDA 11.8 环境下正常运行）。

```Python
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

## **03 模型与数据集下载**

本章节详细介绍如何获取 GR00T-N1.5-3B 预训练模型及相关数据集，为后续的模型部署和微调做好准备。

### 3.1 环境准备

首先需要安装必要的工具并配置下载环境：

```Bash
# 安装 HuggingFace Hub 命令行工具，用于下载模型和数据集
pip install huggingface-hub
```

### 3.2 网络优化配置

为提升下载速度，配置镜像源：

```Bash
# 设置 HuggingFace 镜像端点，加速下载过程
export HF_ENDPOINT=https://hf-mirror.com
```

### 3.3 模型下载

GR00T-N1.5-3B 是 NVIDIA 官方发布的基础模型，包含约 30 亿参数，专门针对人形机器人控制任务优化：

```Bash
# 下载 GR00T-N1.5-3B 模型文件
# --resume-download: 支持断点续传，网络不稳定时可自动恢复下载
# --local-dir: 指定模型保存的本地目录，便于后续管理
huggingface-cli download nvidia/GR00T-N1.5-3B --resume-download --local-dir ./gr00t-model
```

**下载说明：**

- 模型大小约 5.4GB，下载时间视网络状况而定
- 下载完成后模型将保存在 `./gr00t-model` 目录
- 包含完整的模型权重、配置文件及 tokenizer

### 3.4 数据集下载（可选）

如需进行模型微调或验证，可以下载官方提供的数据集：

```Bash
# 下载数据集示例命令
# --repo-type dataset: 明确指定下载类型为数据集
# --resume-download: 同样支持断点续传
# --local-dir: 指定数据集保存路径
# <你的路径>: 替换为实际保存目录，如 ./datasets
# <仓库id>: 替换为目标数据集ID
huggingface-cli download --repo-type dataset --resume-download --local-dir <你的路径> <仓库id>
```

**常用数据集示例：**

```Bash
# 下载 G1 机械臂积木堆叠数据集
huggingface-cli download --repo-type dataset --resume-download --local-dir ./datasets unitreerobotics/G1_BlockStacking_Dataset

# 下载 GR00T 遥操作数据集
huggingface-cli download --repo-type dataset --resume-download --local-dir ./datasets nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1
```

### 3.5 验证下载结果

下载完成后，验证文件完整性：

```Bash
# 检查模型文件
ls -la ./gr00t-model/

# 验证关键文件是否存在
find ./gr00t-model -name "*.safetensors" | wc -l
find ./gr00t-model -name "config.json" -o -name "tokenizer.json"
```

至此，模型和数据集下载完成，可以继续进行环境配置和模型推理步骤。

## **04 数据准备**

### 1、GR00T数据集

GR00T 数据集在设计上并非一个全新的标准，而是 LeRobot V2.0 数据格式的兼容性超集与功能增强版。这意味着：

**核心兼容性：无缝衔接**

- 为 LeRobot V2.0 构建的数据集可以无需修改地在 GR00T 训练流程中直接使用。
- 所有 LeRobot 的核心数据文件（如 `meta/episodes.jsonl`, `meta/tasks.jsonl`, `data/*.parquet`, `videos/*.mp4`）以及关键数据字段（如 `observation.state`, `action`, `timestamp`）都得到完全支持。

**主要功能增强体现在以下四个方面：**

**1. 引入数据模态描述文件 (**`modality.json`**)**

- **GR00T 新增**：要求一个名为 `meta/modality.json` 的配置文件。
- **作用**：该文件像一个“数据说明书”，详细定义了状态和动作数组的语义结构（如哪些维度对应机械臂、哪些对应腿部）、数据的物理含义（如旋转是用四元数还是轴角表示）以及归一化方式。
- **对 LeRobot 数据的影响**：LeRobot 数据集本身不包含此文件。在 GR00T 中若缺失该文件，系统仍能正常运行，但会无法启用依赖精确语义解析的高级功能（如按身体部位进行控制、自动数据规范化等）。

**2. 强化注释系统与多任务支持**

- **LeRobot 基础**：支持基本的任务标识（如 `task_index`）。
- **GR00T 扩展**：提供了一个更丰富、更灵活的注释框架，支持多来源、多类型的标注（例如，同时包含人类语言指令、成功标签、难度评级等）。这些扩展注释遵循 `annotation.<来源>.<类型>` 的命名规则。
- **对 LeRobot 数据的影响**：LeRobot 数据集可以享受基础的训练功能。若要利用 GR00T 更强大的多任务学习与条件化策略能力，则需要按照其扩展规则在数据集中补充相应的注释字段。

**3. 规范化的几何表示与数据处理**

- **GR00T 增强**：明确支持并规范了多种机器人学中常用的旋转表示方法（如四元数、轴角、6D旋转等），并在 `modality.json` 中声明。
- **对 LeRobot 数据的影响**：使用常见格式（如欧拉角或四元数）的 LeRobot 数据完全兼容。但只有按照 GR00T 规范明确定义了旋转格式，才能激活其内部更精确的几何计算、数据增强和训练稳定性优化。

**4. 强制性的本体感知状态要求**

- **GR00T 要求**：明确要求每条数据记录必须包含完整的本体感知状态，即 `observation.state` 字段需要全面覆盖机器人的所有关节传感信息。
- **设计目的**：这一要求旨在提升训练出的模型对于不同机器人硬件平台的泛化能力，确保模型学习到的是基于完整物理状态的、可迁移的决策逻辑。
- **对 LeRobot 数据的影响**：一些较为精简的 LeRobot 数据集可能需要补全其状态观测信息，以满足这一要求，从而充分发挥 GR00T 在大规模跨平台预训练中的潜力。

### 2、modality.json 文件详解

**文件作用与重要性：**

`meta/modality.json` 是GR00T数据集的核心配置文件，它为数据集中的多模态数据提供了语义解析规则。该文件的主要作用包括：

- **语义标注**：将原始的数值数组转换为具有物理意义的机器人部件状态
- **跨平台兼容**：统一不同机器人硬件的数据表示方式
- **自动化处理**：支持模型自动解析和处理复杂的多模态数据
- **扩展性支持**：为未来新增传感器和注释类型提供框架

#### 2.1 state字段 - 状态观测语义映射

`state`字段定义了`observation.state`向量的语义分段，将28维的状态向量按机器人身体部位进行划分：

```JSON
"state": {
    "left_arm": {"start": 0, "end": 7},      // 左臂状态：维度0-6 (7维)
    "right_arm": {"start": 7, "end": 14},    // 右臂状态：维度7-13 (7维)  
    "left_hand": {"start": 14, "end": 21},   // 左手状态：维度14-20 (7维)
    "right_hand": {"start": 21, "end": 28}   // 右手状态：维度21-27 (7维)
}
```

#### 2.2 action字段 - 动作控制语义映射

`action`字段与`state`字段保持完全一致的结构，确保感知与控制的对称性：

```JSON
"action": {
    "left_arm": {"start": 0, "end": 7},      // 左臂控制指令：维度0-6
    "right_arm": {"start": 7, "end": 14},    // 右臂控制指令：维度7-13
    "left_hand": {"start": 14, "end": 21},   // 左手控制指令：维度14-20
    "right_hand": {"start": 21, "end": 28}   // 右手控制指令：维度21-27
}
```

#### 2.3 video字段 - 多视角视觉数据映射

`video`字段配置了三个不同视角的视觉传感器：

```JSON
"video": {
    "cam_right_high": {
        "original_key": "observation.images.cam_right_high"  // 右侧高位视角
    },
    "cam_left_wrist": {
        "original_key": "observation.images.cam_left_wrist"  // 左侧腕部视角
    },
    "cam_right_wrist": {
        "original_key": "observation.images.cam_right_wrist" // 右侧腕部视角
    }
}
```

#### 2.4 annotation字段 - 任务注释映射

`annotation`字段提供了任务描述信息的映射规则：

```JSON
"annotation": {
    "human.task_description": {
        "original_key": "task_index"  // 关联任务索引
    }
}
```

### 3、加载数据集

GR00T提供的官方数据集`demo_data`文件结构如下：

```Python
├── data
│   └── chunk-000
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       ├── episode_000002.parquet
│       ├── episode_000003.parquet
│       └── episode_000004.parquet
├── meta
│   ├── episodes.jsonl
│   ├── info.json
│   ├── modality.json
│   ├── stats.json
│   └── tasks.jsonl
└── videos
    └── chunk-000
        └── observation.images.ego_view
```

运行命令加载数据集：

```Python
python scripts/load_dataset.py --dataset-path ./demo_data/robot_sim.PickNPlace
```

在 NVIDIA RTX 4090 上运行时，遇到了 `FFmpeg` 与 `torchcodec` 版本不兼容的问题：

```Python
RuntimeError: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6 and 7.
          2. The PyTorch version (2.5.1+cu124) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.
```

**问题根源：**

- 系统默认安装的 `FFmpeg` 版本过旧，无法满足 `torchcodec` 的依赖要求
- PyTorch 2.5 版本对 `torchcodec` 有特定的版本兼容性要求

**解决方案：**

- 升级 `FFmpeg`：安装 7.1.1 版本以提供必要的视频解码功能

```Python
# 卸载 FFmpeg
conda uninstall ffmpeg -y

# 安装 FFmpeg 7.1.1
conda install -c conda-forge ffmpeg=7.1.1 -y

# 验证安装
ffmpeg -version
```

- 版本降级：将 `torchcodec` 调整为 0.1 版本，确保与 PyTorch 2.5 的完全兼容

```Python
# 卸载当前版本的 torchcodec
pip uninstall torchcodec -y

# 安装 torchcodec 0.1（与 PyTorch 2.5 兼容）
pip install torchcodec==0.1

# 或者从源码安装（如果 pip 安装有问题）
pip install git+https://github.com/pytorch/torchcodec@v0.1.0
```

这个版本组合经过实际验证，能够有效解决视频解码相关的依赖冲突问题。

### 4、模型微调

#### 4.1 官方教程

用户可运行下方提供的微调脚本，使用示例数据集对 GR00T 模型进行微调。详细的微调流程与操作说明可参考官方文档中的教程：

https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/2_finetuning.ipynb

#### 4.2 适配宇树G1+因时灵巧手

为支持宇树 G1 人形机器人搭配因时（Inspire）灵巧手的硬件配置，需在 `data_config.py` 中新增相应的数据配置类。

随后，在 `DATA_CONFIG_MAP` 中注册该配置：

```Python
"unitree_g1_inspire_2cameras":UnitreeG1InspireData_2cameras_Config(),
```

完成配置后，即可使用如下命令启动微调过程：

```Python
python scripts/gr00t_finetune.py \
   --dataset-path ./datasets/put_house/ \
   --num-gpus 2 \
   --batch-size 4 \
   --output-dir /home/xc/big_disk/model/unitree_g1_inspire_2cameras-checkpoints \
   --max-steps 10000 \
   --data-config unitree_g1_inspire_2cameras \
   --video-backend torchvision_av \
   --embodiment-tag new_embodiment \
   --base-model-path /home/xc/gr00t-model \
   --no-tune_diffusion_model   
```

通过以上步骤，用户可针对宇树 G1 与因时灵巧手的组合硬件平台，灵活定制并完成 GR00T 模型的微调任务，从而更好地适配实际场景中的感知与控制需求。

# GR00T真机部署流程

## 01 宇树PC2环境配置

### 1.1 image_server服务部署

将`image_server.py`代码复制到宇树PC2上

### 1.2 inspire手部服务部署

参考[GitHub - NaCl-1374/inspire_hand_ws](https://github.com/NaCl-1374/inspire_hand_ws)，在PC2上进行部署

```Python
# 克隆手部控制仓库
git clone https://github.com/NaCl-1374/inspire_hand_ws

# 安装Python依赖
pip install -r requirements.txt

# 初始化子模块git submodule init
git submodule update

# 安装Unitree SDK
cd unitree_sdk2_python
pip install -e .

# 安装Inspire Hand SDK  
cd ../inspire_hand_sdk
pip install -e .
```

## 02 真机部署

### 2.1 启动模型服务端

```Python
python scripts/inference_service.py \
--server \
--http-server \
--model-path /home/xc/big_disk/model/unitree_g1_inspire_2cameras-checkpoints/checkpoint-10000 \
--embodiment-tag new_embodiment \
--data-config unitree_g1_inspire_2cameras \
--port 6666 \
--host 0.0.0.0
```

### 3.2 启动图像和手部服务

宇树PC2上启动图像服务

```Python
python image_server.py
```

宇树PC2上启动手部服务

```Python
python inspire_hand_ws/inspire_hand_sdk/example/Headless_driver_double.py
```

### 3.3 启动模型客户端

```Python
python g1_inspire.py
```