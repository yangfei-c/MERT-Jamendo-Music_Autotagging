import numpy as np
import os

TARGET_TAGS = {
    # [流派 Genres - 10个] - 常见音乐流派，覆盖流行到古典
    'genres': [
        'pop',
        'electronic',
        'rock',
        'classical',
        'jazz',
        'hiphop',
        'ambient',
        'folk',
        'metal',
        'soundtrack'
    ],

    # [乐器 Instruments - 7个] - 覆盖：四大件 + 电子 + 管弦 + 人声
    'instruments': [
        'piano',  # 键盘类 - 原声
        'synthesizer',  # 键盘类 - 电子
        'guitar',  # 弦乐类 - 拨弦
        'strings',  # 弦乐类 - 拉弦 (大提琴/小提琴等)
        'drums',  # 打击类
        'bass',  # 低音类
        'voice'  # 人声 (区分纯音乐的关键)
    ],

    # [情绪 Moods/Themes - 9个] - 基于情感象限分类，创新添加分类标签以便未来扩展
    'moods/themes': [
        'happy',  # [Pos/High] 快乐
        'energetic',  # [Pos/High] 活力/运动
        'relaxing',  # [Pos/Low]  放松/助眠
        'dream',  # [Pos/Low]  梦幻/迷幻 (替换了之前的 inspiring，dream 的听感更独特)
        'sad',  # [Neg/Low]  悲伤/治愈
        'dark',  # [Neg/High] 暗黑/压抑
        'epic',  # [High]     史诗/震撼
        'film',  # [Scene]    电影感/画面感
        'love'  # [Topic]    爱情/浪漫
    ]
}

SAVE_DIR = os.environ.get('SAVE_DIR', r"D:\AAAA_audio_dataset\mtg-jamendo\meta")
SAVE_PATH = os.path.join(SAVE_DIR, "tag_recsys.npy")


def get_recsys_npy():
    formatted_tags = []
    print("Generating Tag Taxonomy...")  # 创新：更详细的打印信息

    # 创新：使用字典映射前缀，避免硬编码 if-elif，提高扩展性
    prefix_map = {
        'genres': "genre---",
        'instruments': "instrument---",
        'moods/themes': "mood/theme---"
    }

    for category, tags in TARGET_TAGS.items():
        prefix = prefix_map.get(category, "")
        for t in tags:
            full_tag = f"{prefix}{t}"
            formatted_tags.append(full_tag)
            print(f"Added: {full_tag}")  # 创新：逐个打印添加的标签，便于调试

    formatted_tags = np.array(sorted(formatted_tags))

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    np.save(SAVE_PATH, formatted_tags)

    print(f"✅ 处理完成！Saved to {SAVE_PATH}")

    # 加载并打印以验证
    da = np.load(SAVE_PATH)
    print("Generated Tags:")
    print(da)


if __name__ == "__main__":
    get_recsys_npy()