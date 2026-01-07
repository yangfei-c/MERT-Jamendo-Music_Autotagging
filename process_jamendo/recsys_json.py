import json
import os
import numpy as np
from collections import Counter
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 黄金标签文件
TAG_NPY_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\meta\tag_recsys.npy"

# 2. 元数据文件 (查作者、歌名用)
META_JSON_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\mtg_meta.json"

# 3. 三个标签源文件 (提供准确的 genre/instrument/mood)
SOURCE_FILES = [
    r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON\autotagging-train.json",
    r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON\autotagging-validation.json",
    r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON\autotagging-test.json"
]

# 4. 输出文件
OUTPUT_JSON_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON\recsys.json"


# ===========================================

def generate_recsys_dataset():
    # --- 1. 加载黄金标签 ---
    print(f"📖 加载筛选标签: {TAG_NPY_PATH}")
    if not os.path.exists(TAG_NPY_PATH):
        print("❌ 错误: 找不到 tag_recsys.npy")
        return
    target_tags_set = set(np.load(TAG_NPY_PATH))
    print(f"✅ 已加载 {len(target_tags_set)} 个目标标签。")

    # --- 2. 加载元数据 (构建查询字典) ---
    print(f"📖 加载元数据 (用于补充歌名/作者): {META_JSON_PATH}")
    meta_dict = {}
    if os.path.exists(META_JSON_PATH):
        with open(META_JSON_PATH, 'r', encoding='utf-8') as f:
            meta_list = json.load(f)
            # 将列表转换为字典，key 是 track_id，方便快速查找
            for item in meta_list:
                tid = item.get('track_id')
                if tid:
                    meta_dict[tid] = item
        print(f"✅ 元数据加载完成，共 {len(meta_dict)} 首歌曲信息。")
    else:
        print("⚠️ 警告: 找不到 mtg_meta.json，生成的文件的 track_artist 等字段将为空！")

    # --- 3. 遍历三个文件进行过滤与合并 ---
    filtered_tracks = []
    tag_counter = Counter()

    # 映射关系: autotagging文件中的复数键 -> 黄金标签的前缀
    category_map = {
        'genres': 'genre---',
        'instruments': 'instrument---',
        'moods/themes': 'mood/theme---'
    }

    # 输出时的单数键名 (为了符合您的图片格式)
    output_keys = {
        'genres': 'genre',
        'instruments': 'instrument',
        'moods/themes': 'mood/theme'
    }

    print("🚀 开始处理三个源文件...")

    for src_file in SOURCE_FILES:
        if not os.path.exists(src_file):
            print(f"⚠️ 跳过不存在的文件: {src_file}")
            continue

        print(f"   -> 处理: {os.path.basename(src_file)}")
        with open(src_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        for item in tqdm(source_data, desc="Filtering"):
            track_id = item.get('id')

            # 临时存放匹配到的标签
            matched_tags = {
                'genre': [],
                'instrument': [],
                'mood/theme': []
            }
            has_match = False

            # 检查标签
            for src_key, prefix in category_map.items():
                out_key = output_keys[src_key]
                raw_tags = item.get(src_key, [])

                for t in raw_tags:
                    full_tag = f"{prefix}{t}"
                    if full_tag in target_tags_set:
                        matched_tags[out_key].append(t)
                        tag_counter[full_tag] += 1
                        has_match = True

            # 如果命中目标标签
            if has_match:
                # 尝试从元数据字典里获取详细信息
                meta_info = meta_dict.get(track_id, {})

                # 构建最终对象 (字段对齐您的要求)
                new_track = {
                    "track_id": track_id,
                    # 优先用 meta 里的信息，如果没有则给默认值
                    "track_artist": meta_info.get("track_artist") or meta_info.get("artist_name", "Unknown"),
                    "track_name": meta_info.get("track_name", "Unknown"),
                    "track_album": meta_info.get("track_album") or meta_info.get("album_name", ""),
                    "track_year": meta_info.get("track_year") or meta_info.get("album_date", ""),
                    "track_url": meta_info.get("track_url") or meta_info.get("audio_url", ""),

                    # 标签部分 (单数键名)
                    "genre": matched_tags['genre'],
                    "instrument": matched_tags['instrument'],
                    "mood/theme": matched_tags['mood/theme'],

                    "track_duration": meta_info.get("track_duration") or meta_info.get("duration", ""),

                    # 路径：优先保留带后缀的 meta path
                    "track_path": meta_info.get("track_path") or meta_info.get("path") or item.get("path")
                }

                filtered_tracks.append(new_track)

    # --- 4. 保存结果 ---
    print(f"\n💾 正在保存 {len(filtered_tracks)} 条数据到 {OUTPUT_JSON_PATH}...")

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_tracks, f, ensure_ascii=False, indent=4)

    # --- 5. 打印统计 (按字母顺序排序，即按类别分组) ---
    print("\n" + "=" * 60)
    print("📊 最终数据集统计 (Sorted by Category)")
    print("=" * 60)

    # 修改核心：key=lambda x: x[0] 表示按标签名排序
    # 因为标签名是 genre---abc, instrument---xyz, 所以会自动归类
    sorted_tags = sorted(tag_counter.items(), key=lambda x: x[0])

    current_prefix = ""
    for tag, count in sorted_tags:
        # 为了更美观，当类别变化时打印一个空行（可选）
        prefix = tag.split('---')[0]
        if prefix != current_prefix:
            if current_prefix != "": print("-" * 40)
            current_prefix = prefix

        print(f"{tag:<40} {count}")

    print("=" * 60)
    print(f"🎉 全部完成！总共筛选出 {len(filtered_tracks)} 首歌曲。")


if __name__ == "__main__":
    generate_recsys_dataset()