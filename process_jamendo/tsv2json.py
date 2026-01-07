import csv
import json
import os
from collections import Counter

# --- TSV 转 JSON  ---
def tsv2json(input_file, output_file):
    data_list = []
    if not os.path.exists(input_file):
        print(f"找不到输入文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            header = next(reader)
        except StopIteration:
            return

        for row in reader:
            if not row or len(row) < 5: continue
            track_id = row[0]

            path = os.path.splitext(row[3])[0]
            item = {"id": track_id, "path": path, "genres": [], "instruments": [], "moods/themes": []}
            for cell in row[5:]:
                cell = cell.strip()
                if not cell: continue
                if "---" in cell:
                    tag_parts = cell.split("---", 1)
                    tag_type, tag_value = tag_parts[0].strip().lower(), tag_parts[1].strip()
                    if tag_type == "genre":
                        item["genres"].append(tag_value)
                    elif tag_type == "instrument":
                        item["instruments"].append(tag_value)
                    elif tag_type == "mood/theme":
                        item["moods/themes"].append(tag_value)
            data_list.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    print(f"转换成功: {os.path.basename(output_file)} (包含 {len(data_list)} 条)")


# --- analyze json ---
def analyze(json_files):
    genre_counts = Counter()
    instr_counts = Counter()
    mood_counts = Counter()
    total_tracks = 0

    print("\n" + "=" * 60)
    print("开始统计标签频次（train/test/validation）...")

    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"文件不存在，跳过统计 {json_file}")
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            total_tracks += len(data)
            for item in data:
                genre_counts.update(item.get('genres', []))
                instr_counts.update(item.get('instruments', []))
                mood_counts.update(item.get('moods/themes', []))

    print("=" * 60)
    print("           标签频次统计报告")
    print("=" * 60)
    print(f"涵盖文件: {[os.path.basename(f) for f in json_files]}")
    print(f"累计总曲目数: {total_tracks}")
    print("-" * 60)

    # 辅助打印函数：展示种类数、总次数以及每个标签的频次
    def print_tag_detail(name, counter_obj):
        # unique_count 就是 Counter 的长度
        unique_count = len(counter_obj)
        total_instances = sum(counter_obj.values())

        print(f"[{name}]")
        print(f" -> 不同种类: {unique_count} 种")
        print(f" -> 标注总数: {total_instances} 次")
        print(f" -> 各标签频次 (按从高到低排序):")

        # 获取排序后的结果
        sorted_list = counter_obj.most_common()

        # 每行打印 3 个标签及其次数，节省空间
        for i in range(0, len(sorted_list), 3):
            line_items = [f"{tag}: {count}" for tag, count in sorted_list[i:i + 3]]
            print("    " + " | ".join(line_items))
        print("-" * 60)

    print_tag_detail("流派 Genres", genre_counts)
    print_tag_detail("乐器 Instruments", instr_counts)
    print_tag_detail("情绪/主题 Moods/Themes", mood_counts)

    print("汇总任务完成。")


# --- 关键配置与执行区 ---
base_path = r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0"
splits = ["train", "test", "validation"]

json_path_list = []

# 先进行转换
for s in splits:
    in_tsv = os.path.join(base_path, rf"TSV\autotagging-{s}.tsv")
    out_json = os.path.join(base_path, rf"JSON\autotagging-{s}.json")
    tsv2json(in_tsv, out_json)
    json_path_list.append(out_json)
# 进行分析标签
analyze(json_path_list)