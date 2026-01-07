import os

# 配置目标文件夹路径
TARGET_DIR = r"D:\AAAA_audio_dataset\mtg-jamendo\audio"


def remove_low_suffix():
    print(f"[*] 正在扫描文件夹: {TARGET_DIR} ...")

    if not os.path.exists(TARGET_DIR):
        print(f"[!] 错误: 找不到路径 {TARGET_DIR}")
        return

    count = 0
    skipped = 0

    # os.walk 会自动遍历所有子文件夹 (包括 00, 01 ... 99)
    for root, dirs, files in os.walk(TARGET_DIR):
        for filename in files:
            # 只处理包含 .low 的文件
            if ".low" in filename:
                old_path = os.path.join(root, filename)

                # 生成新名字: 12345.low.mp3 -> 12345.mp3
                new_filename = filename.replace(".low", "")
                new_path = os.path.join(root, new_filename)

                try:
                    # 检查目标文件是否已经存在 (防止冲突)
                    if os.path.exists(new_path):
                        print(f"[跳过] 目标已存在: {new_filename}")
                        skipped += 1
                        continue

                    # 执行重命名
                    os.rename(old_path, new_path)
                    count += 1

                    # 每改100个打印一次进度，防止刷屏太快
                    if count % 100 == 0:
                        print(f"[*] 已处理 {count} 个文件... (当前: {new_filename})")

                except Exception as e:
                    print(f"[!] 重命名失败 {filename}: {e}")

    print("-" * 30)
    print(f"处理完成！")
    print(f"成功重命名: {count} 个")
    print(f"跳过(已存在): {skipped} 个")
    print("-" * 30)


if __name__ == "__main__":
    remove_low_suffix()