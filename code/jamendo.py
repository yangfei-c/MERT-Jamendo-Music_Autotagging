import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig


class JamendoDataset(Dataset):
    def __init__(self, config: DictConfig, split: str = 'train'):
        self.cfg = config
        self.split = split
        self.data = self._load_data()
        self.tag_to_idx = self._load_tags()
        self.num_classes = len(self.tag_to_idx)
        print(f"[{split.upper()}] Task:[{self.cfg.task_name}] Samples: {len(self.data)}, Classes: {self.num_classes}")

    def _load_data(self):
        """
        内部函数：根据任务名和 split 自动构建路径并加载 JSON 数据
        """
        split_dir = self.cfg.data.split_dir
        task = self.cfg.task_name
        #根据task获取前缀
        if task.lower() == 'autotagging':
            prefix = "autotagging"
        else:
            prefix = f"autotagging_{task}"
        #拼接完整，判断是否存在并加载
        json_filename = f"{prefix}-{self.split}.json"
        json_path = os.path.join(split_dir, json_filename)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到数据集文件: {json_path}\n当前任务: {task}, 尝试读取: {json_filename}")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    def _load_tags(self):
        """
        内部函数加载标签
        """
        tag_path = self.cfg.data.tag_list_path
        if not os.path.exists(tag_path):
            raise FileNotFoundError(f"标签表不存在: {tag_path}！")
        task = self.cfg.task_name.lower()
        filtered_tags = []

        if task == 'top50tags':
            target_path = self.cfg.data.tag_list_50_path
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Top50 标签文件未找到: {target_path}")
            filtered_tags = np.load(target_path)
        else:
            # 只有不是 Top50 时，才需要加载这个大文件
            all_tags = np.load(tag_path)
            #根据不同任务获取不同标签
            if task == 'genre':
                filtered_tags = [t for t in all_tags if t.startswith('genre---')]
            elif task == 'instrument':
                filtered_tags = [t for t in all_tags if t.startswith('instrument---')]
            elif task == 'moodtheme':
                filtered_tags = [t for t in all_tags if t.startswith('mood/theme---')]
            else:
                #autotagging
                filtered_tags = all_tags
        #print(f"   [Tag Info] Task: {task} | Loaded {len(filtered_tags)} tags.")
        return {tag: i for i, tag in enumerate(filtered_tags)}
    def _format_tag(self, tag_raw, tag_json):
        # tag_list里面内容是genre---rock这种，json中是genres[rock,blues]这种
        if tag_json == 'genres': return f"genre---{tag_raw}"
        if tag_json == 'instruments': return f"instrument---{tag_raw}"
        if tag_json == 'moods/themes': return f"mood/theme---{tag_raw}"
        return tag_raw
    def _get_labels(self, item_dict):
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for cat in ['genres', 'instruments', 'moods/themes']:
            for t in item_dict.get(cat, []):
                full_tag = self._format_tag(t, cat)
                if full_tag in self.tag_to_idx:
                    label[self.tag_to_idx[full_tag]] = 1.0
        return label
    def _load_mert_feature(self, path_str):
        path_str = path_str.replace('/', os.sep)
        track_dir = os.path.join(self.cfg.data.feature_dir, path_str)

        if not os.path.exists(track_dir):
            return torch.zeros(self.cfg.data.input_dim)

        segments = [f for f in os.listdir(track_dir) if f.endswith('.npy')]
        if not segments:
            return torch.zeros(self.cfg.data.input_dim)

        selected = random.choice(segments) if self.split == 'train' else segments[0]
        try:
            feat = np.load(os.path.join(track_dir, selected))
            feat = torch.from_numpy(feat).float()
            if feat.dim() > 1: feat = feat.mean(dim=0)
            return feat
        except:
            return torch.zeros(self.cfg.data.input_dim)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        x = self._load_mert_feature(item['path'])
        y = self._get_labels(item)
        return x, y