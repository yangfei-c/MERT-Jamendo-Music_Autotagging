import os
import requests
import tarfile
from tqdm import tqdm
import time

# ================= 配置区域 =================

# 1. 代理设置
PROXY_URL = "http://127.0.0.1:7890"

# 2. 检查和存放文件夹的根路径 (对应你说的 D:\AAAA_audio_dataset\mtg-jamendo\audio)
# 脚本会在这里直接找名为 "00", "01" 的文件夹
BASE_PATH = r"D:\AAAA_audio_dataset\mtg-jamendo\audio"

# 3. 下载源地址
SOURCE_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ===========================================

def setup_proxy():
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL
    print(f"[*] 代理已设置为: {PROXY_URL}")


def get_batch_id(filename):
    """从文件名 raw_30s_audio-low-00.tar 提取出 '00'"""
    try:
        return filename.replace('.tar', '').split('-')[-1]
    except:
        return None


def download_file(url, filepath):
    """支持断点续传的下载"""
    resume_header = {}
    file_mode = 'wb'
    downloaded_bytes = 0

    if os.path.exists(filepath):
        downloaded_bytes = os.path.getsize(filepath)
        resume_header = {'Range': f'bytes={downloaded_bytes}-'}
        file_mode = 'ab'
        print(
            f"[*] 发现未完成的压缩包，断点续传: {os.path.basename(filepath)} (已完成 {downloaded_bytes / (1024 * 1024):.2f} MB)")

    try:
        response = requests.get(url, headers={**HEADERS, **resume_header}, stream=True, timeout=60)
        if response.status_code == 416: return True
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0)) + downloaded_bytes
        with open(filepath, file_mode) as f, tqdm(
                desc=os.path.basename(filepath),
                initial=downloaded_bytes,
                total=total_size,
                unit='B', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"\n[!] 下载出错: {e}")
        return False


def main():
    setup_proxy()

    # 确保基础路径存在
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    # 循环 00 到 99
    for i in range(100):
        batch_id = f"{i:02d}"  # 生成 "00", "01", "02" ...

        # --- 核心逻辑：只检查文件夹是否存在 ---
        # 路径: D:\AAAA_audio_dataset\mtg-jamendo\audio\00
        check_folder_path = os.path.join(BASE_PATH, batch_id)

        if os.path.exists(check_folder_path) and os.path.isdir(check_folder_path):
            print(f"[√] 检测到文件夹 {batch_id} 已存在，跳过该任务。")
            continue
        # ------------------------------------

        # 如果文件夹不存在，则构建下载信息
        file_name = f"raw_30s_audio-low-{batch_id}.tar"
        file_url = SOURCE_URL + file_name
        local_tar_path = os.path.join(BASE_PATH, file_name)

        print(f"\n[*] 文件夹 {batch_id} 不存在，开始处理...")

        # 尝试下载
        success = False
        for retry in range(3):
            if download_file(file_url, local_tar_path):
                success = True
                break
            else:
                print(f"[!] 下载失败，5秒后进行第 {retry + 1} 次重试...")
                time.sleep(5)

        if success:
            print(f"[*] 正在解压 {file_name} 到 {BASE_PATH}...")
            try:
                with tarfile.open(local_tar_path, "r:") as tar:
                    tar.extractall(BASE_PATH)
                # 解压成功后删除压缩包
                os.remove(local_tar_path)
                print(f"[√] 编号 {batch_id} 处理完成。")
            except Exception as e:
                print(f"[!] 解压失败: {e}")
        else:
            print(f"[x] 编号 {batch_id} 经过多次尝试后下载失败。")


if __name__ == "__main__":
    main()