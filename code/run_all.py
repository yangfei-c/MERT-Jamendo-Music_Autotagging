import subprocess
import sys
import time


def run_tasks():
    tasks = ["Genre", "Instrument", "MoodTheme", "Top50tags", "Autotagging"]
    model_type = "Linear"
    # 记录开始时间
    total_start_time = time.time()
    print(f"🚀 开始批量运行任务... 模型类型: {model_type}")
    for task in tasks:
        print("\n" + "=" * 60)
        print(f"▶️  正在启动任务: {task}")
        print("=" * 60)
        cmd = [
            sys.executable, "train.py",
            f"task_name={task}",
            f"model_type={model_type}"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✅ 任务 {task} 完成！")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 任务 {task} 运行失败！错误代码: {e.returncode}")
            continue
        except KeyboardInterrupt:
            print("\n🛑 用户手动停止脚本。")
            sys.exit(0)

    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"🎉 所有任务处理完毕！总耗时: {total_time / 60:.2f} 分钟")
    print("=" * 60)

if __name__ == "__main__":
    run_tasks()