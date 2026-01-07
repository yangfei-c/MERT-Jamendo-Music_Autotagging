# Music Auto-Tagging With Mert 🎵

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)
![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd.svg)
![MERT](https://img.shields.io/badge/Model-MERT-green)

这是一个基于 MERT (Music undERstanding model with large-scale self-supervised Training) 预训练特征的音乐自动标注项目。项目利用 MERT 的强大音频理解能力提取特征，并结合 MLP 分类头，在 MTG-Jamendo 数据集上实现高效音乐标注。

> 项目代码参考了 Gemini 生成的代码及相关论文的开源实现

## 📖 核心特性

* 🚀 **SOTA 特征提取**：基于 Hugging Face 的 MERT-v1-95M 模型，提取 768 维音频 Embedding，捕捉深层声学特征。
* ⚡ **高效轻量化模型**：采用优化的 Linear (MLP) Backbone，结合 BatchNorm 和 Dropout，训练速度快、推理高效。
* ⚖️ **样本不平衡处理**：集成 Asymmetric Loss，有效解决多标签分类中正负样本比例失衡问题。

## 📂 项目结构 

```text
MERT-Jamendo-Music_Autotagging/
├── checkpoints/                # 训练产物
├── code/                       # 核心训练代码
│   ├── config/                 # Hydra 配置文件
│   │   └── config.yaml         # 全局参数配置
│   ├── model/                  # 模型定义
│   │   └── linear.py           # Linear (MLP) Backbone
│   ├── trainer/                # Lightning Module
│   │   └── linear_trainer.py   # 训练逻辑与 Loss 定义
│   ├── jamendo.py              # Dataset 定义
│   ├── data_loader.py          # Lightning DataModule
│   ├── train.py                # 单任务训练入口
│   ├── run_all.py              # 批量任务运行脚本
│   └── visual.py               # 结果可视化与评估指标计算
│
├── process_jamendo/            # 数据预处理流水线
│   ├── dl.py                   # 下载数据集（audio-low），并解压删除压缩包
│   ├── del_.low.py             # 删除解压后数据集音频名中的.low
│   ├── tsv2json.py             # 原始 TSV 转 JSON & 统计标签
│   ├── recsys_npy.py           # 用于推荐的标签列表以及生成npy文件存放
│   ├── recsys_json.py          # 根据recsys.npy构建最终推荐数据集 
│   └── recsys_mert_feature.py  # 提取recsys.json的mert特征（仅仅提取567层，并保留时间维度）
│
├── results/                    # 训练产物
├── tb_logs/                    # 训练产物
├── README.md                   # 项目文档
```

## 🔄 数据集介绍 (MTG-Jamendo)

MTG-Jamendo 是音乐信息检索 (MIR) 领域最大、最开放的音乐自动标注数据集之一，由西班牙庞培法布拉大学 (Universitat Pompeu Fabra) 的音乐技术小组 (MTG) 构建。数据集包含流派 (Genre)、乐器 (Instrument) 和情绪/主题 (Mood/Theme) 等标签，总计 183 类（Genre: 87，Instrument: 40，Mood/Theme: 56）。原始数据存在部分错标和漏标，因此本项目进行了针对性筛选。

采用官方划分 split-0，便于复现和对比。参考仓库：[MTG-Jamendo GitHub](https://github.com/MTG/mtg-jamendo-dataset/tree/master/data/splits/split-0?referrer=grok.com)。

### 1. 下载和处理数据集 

可以去克隆MTG—Jamendo数据集Github官方仓库，然后根据所需下载数据集。该项目通过脚本直接去数据集发布页进行下载，由于个人电脑内存限制，选择下载low版本。
该项目中用于训练的划分，采用MTG—Jamendo官方划分（便于别人复现和对比）
划分选用split-0。

> https://github.com/MTG/mtg-jamendo-dataset/tree/master/data/splits/split-0

- **下载数据集**：运行 `process_jamendo/dl.py`（进行数据集下载，解压以及删除压缩包）
- **去除命名冗余**：运行`process_jamendo/del_.low.py`（删除解压之后音频名中的.low，如214.low.mp3----->214.mp3）
- **格式转换**: 运行 `process_jamendo/tsv2json.py` （TSV数据格式转为JSON格式，并进行分析）
  - **genre**（**87**类）：Electronic (16480)、Soundtrack (8094)、Pop (7805)、Ambient (7570)、Rock (6865)、Classical (5602) 等。
  - **Instrument (40 类)**：Piano (7750)、Synthesizer (7496)、Drums (6120)、Bass (5726)、Guitar (4804)、Electric Guitar (4241) 等。
  - **Mood/Theme (56 类)**：Happy (1657)、Film (1502)、Energetic (1357)、Relaxing (1350)、Emotional (1271)、Melodic (1213)、Dark (1202)、Epic (982)、Dream (951) 等。

- **标签筛选** : 运行 `process_jamendo/tags_recsys.py`，筛选 26 个核心标签，生成 tag_recsys.npy 和 recsys.json）。筛选标准：保留高语义价值标签，适用于音乐推荐。

  - **genre (10类)**: Pop, Rock, Electronic, Jazz 等。
  - **Instruments (7类)**: Piano, Guitar, Drums, Voice 等。
  - **Moods/Themes (9类)**: Happy, Sad, Energetic, Dark 等。

- **标签查看**： 

  `process_jamendo/tags_list.py`，挑选的26个标签

  `process_jamendo/generate_npy.py`，存放的26个挑选

## 📊 实验任务

本项目在 **MTG-Jamendo** 数据集上执行了以下分类任务实验。模型均采用 MERT 特征作为输入。

### 1. 基础任务

| 任务名称   | 类别数  | 描述                                      |
| ---------- | ------- | ----------------------------------------- |
| **Top-50** | 50      | 仅对出现频率最高的 50 个标签分类。            |
| **Mood/Theme** | 56      | 情绪/主题分类，如 Happy, Sad, Film, Epic 等。 |
| **Instrument** | 40      | 乐器分类，如 Guitar, Piano, Strings 等。      |
| **Genre**  | 87      | 流派分类，如 Rock, Pop, Jazz, Electronic 等。 |
| **All**    | 183     | 所有标签分类。                                |

### 2. 推荐任务 

为了后续的音乐推荐系统，特别设计了 **RecSys Tags** 任务。

> **总类**：26 类（10 Genres + 7 Instruments + 9 Moods）。
>
> **目标**：构建清晰的用户画像，支持推荐应用。

## 📈 结果展示 (Results)

*注：以下展示为部分任务在测试集上的性能可视化（示例图占位，待训练完成后更新具体数据）。*

### ROC 曲线对比 (All Tasks)

下图展示了不同任务（Genre, Instrument, Mood）在测试集上的 ROC 曲线表现。曲线越靠近左上角，说明模型区分正负样本的能力越强。

### 精确率-召回率曲线 (Precision-Recall Curve)

对于多标签不平衡数据，PR 曲线更能反映模型在正样本（即实际存在的标签）上的预测准确度。