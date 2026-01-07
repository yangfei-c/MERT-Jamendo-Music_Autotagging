import json
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 您刚刚生成的 JSON 文件路径
JSON_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON\recsys.json"

# 2. 音频文件的主根目录
# 脚本会将 JSON 里的相对路径拼接在这个目录后面
AUDIO_BASE_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\audio"

# 3. 如果有文件丢失，将丢失列表保存到这里，方便后续处理
MISSING_LOG_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\splits\split-0\JSON\missing_files.txt"


# ===========================================

def check_files():
    print(f"📖 正在读取列表: {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        print("❌ 错误: 找不到 recsys.json 文件")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = len(data)
    print(f"✅ 加载成功，共需检查 {total_count} 首音乐。\n")

    found_count = 0
    missing_count = 0
    missing_list = []

    print(f"🚀 开始扫描音频文件 (根目录: {AUDIO_BASE_DIR})...")

    # 使用 tqdm 显示进度条
    for item in tqdm(data, desc="Checking"):
        track_id = item.get('track_id', 'Unknown')
        rel_path = item.get('track_path')

        if not rel_path:
            # 如果 json 里连路径字段都没有
            missing_count += 1
            missing_list.append(f"[ID: {track_id}] ❌ JSON中缺少 track_path 字段")
            continue

        # === 核心逻辑：拼接路径 ===
        # 1. 规范化路径分隔符 (把 json 里的 / 变成 windows 的 \)
        rel_path = rel_path.replace('/', os.sep)

        # 2. 拼接完整路径
        full_path = os.path.join(AUDIO_BASE_DIR, rel_path)

        # 3. 检查文件是否存在
        if os.path.exists(full_path):
            found_count += 1
        else:
            missing_count += 1
            # 记录详细的丢失信息
            missing_list.append(f"[ID: {track_id}] ❌ 丢失: {full_path}")

    # === 打印报告 ===
    print("\n" + "=" * 60)
    print("📊 音频文件体检报告")
    print("=" * 60)
    print(f"总数 (Total):   {total_count}")
    print(f"存在 (Found):   {found_count}  (✅ 正常)")
    print(f"缺失 (Missing): {missing_count}  (❌ 异常)")
    print("=" * 60)

    # 如果有缺失，保存日志
    if missing_count > 0:
        print(f"⚠️ 发现 {missing_count} 个文件缺失！详细列表已保存至:")
        print(f"📄 {MISSING_LOG_PATH}")

        with open(MISSING_LOG_PATH, 'w', encoding='utf-8') as f:
            f.write(f"检查时间: {JSON_PATH}\n")
            f.write(f"根目录: {AUDIO_BASE_DIR}\n")
            f.write("=" * 50 + "\n")
            for line in missing_list:
                f.write(line + "\n")

        print("\n💡 建议检查：")
        print("1. 音频文件夹路径是否配置正确？")
        print("2. 是否下载了完整的数据集？")
        print("3. JSON 里的路径是 '00/123.mp3' 还是没有后缀？")
    else:
        print("🎉 完美！所有音频文件都在它们该在的地方！")


if __name__ == "__main__":
    check_files()