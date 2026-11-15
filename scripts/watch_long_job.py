"""长任务看门狗：判定一个 GPU/长时作业是「活着 / 卡住 / 已死」。

背景（2026-11-10 真实事故 T8-R1-A1）：rest_r1 首次训练挂在 teammate 的后台作业下，
成员回合结束时作业被回收，python 子进程被连带杀掉 —— 日志停在 step 25、无报错、无
OOM、adapter 目录为空。事后只能靠「日志不再增长 + GPU 无计算进程」推断，缺少直接依据。

本脚本把这套推断固化，供任何长任务启动前/运行中调用：

    python scripts/watch_long_job.py --log logs/train/rest_r1.jsonl
    python scripts/watch_long_job.py --log <log> --max-idle-seconds 300 --expect-process-substring train_sft

判定规则（全部基于只读探测，不碰被测进程）：
* 日志最后写入距今 > ``--max-idle-seconds`` 且 GPU 无对应 python 计算进程 → ``dead``
* 日志停了但 GPU 仍有计算进程 → ``stalled``（可能在算大 batch / 卡在 IO）
* 日志近期有写入 → ``alive``
* 日志存在且 --expect-finished-marker 出现 → ``finished``

退出码：alive/finished = 0，stalled = 2，dead = 1（便于在流水线里直接判）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def nvidia_smi(*fields: str) -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def compute_processes() -> list[dict[str, str]]:
    """列出真正占用 GPU 的计算进程（python 才算训练/评测）。"""

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"pid": parts[0], "process_name": parts[1], "used_memory": parts[2]})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="被监视作业的日志/进度文件")
    parser.add_argument("--max-idle-seconds", type=float, default=300.0)
    parser.add_argument(
        "--expect-process-substring",
        default="python",
        help="GPU 计算进程名里应出现的子串；默认 python",
    )
    parser.add_argument("--expect-finished-marker", default=None, help="日志里出现该字符串即视为正常结束")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    now = time.time()
    gpu = nvidia_smi("memory.used", "utilization.gpu")
    processes = compute_processes()
    python_on_gpu = [
        row for row in processes if args.expect_process_substring.lower() in row["process_name"].lower()
    ]

    if not log_path.is_file():
        payload = {
            "status": "no_log",
            "log": str(log_path),
            "note": "日志文件不存在：作业可能还没启动，或路径写错",
            "gpu": gpu,
            "compute_processes": processes,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    idle_seconds = now - log_path.stat().st_mtime
    finished = False
    if args.expect_finished_marker:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            finished = args.expect_finished_marker in handle.read()

    if finished:
        status = "finished"
    elif python_on_gpu and idle_seconds > args.max_idle_seconds:
        status = "stalled"
    elif python_on_gpu:
        status = "alive"
    elif idle_seconds > args.max_idle_seconds:
        status = "dead"
    else:
        status = "alive"

    payload = {
        "status": status,
        "log": str(log_path),
        "log_idle_seconds": round(idle_seconds, 1),
        "max_idle_seconds": args.max_idle_seconds,
        "gpu": gpu,
        "python_compute_processes": python_on_gpu,
        "other_compute_processes": [row for row in processes if row not in python_on_gpu],
        "hint": {
            "alive": "正常：日志在增长且有 python 占用 GPU",
            "stalled": "日志停止但进程仍在：可能在算大 batch，或卡在数据 IO/锁",
            "dead": "判定已死：日志长时间不写且 GPU 无 python 计算进程；按 PLAN §10 先取证再重启",
            "finished": "检测到结束标记",
            "no_log": "日志不存在",
        }.get(status, ""),
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return {"alive": 0, "finished": 0, "stalled": 2, "dead": 1, "no_log": 1}[status]


if __name__ == "__main__":
    raise SystemExit(main())
