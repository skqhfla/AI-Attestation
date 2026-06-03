#!/usr/bin/env python3
"""
generate_challenge_bench.py 를 모드(CPU/GPU) × 워커 수(N=1..5) 로 sweep 하면서
RAM(RSS) 과 VRAM 을 시계열로 로깅한다.

각 (mode, N) 조합마다:
  1) 깨끗한 출력 디렉터리에서 N 개 자식 프로세스 동시 실행
  2) interval 초마다 자식별 RSS + 합계 + nvidia-smi VRAM 샘플링
  3) per-run CSV + 마지막에 master summary CSV 저장

사용:
  python sweep_resources.py
    --mode both              (cpu | gpu | both, default: both)
    --max-workers 5          (default: 5; N=1..max_workers 까지)
    --challenges 100         (자식 1개당 생성할 이미지 수, default: 100)
    --model resnet20_quan    (default: resnet20_quan, 모든 워커가 같은 모델)
    --interval 0.5           (샘플링 주기 초, default: 0.5)
    --logdir bench_logs      (default: ./bench_logs/<timestamp>)
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_SCRIPT = SCRIPT_DIR / "generate_challenge_bench.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['cpu', 'gpu', 'both'], default='both')
    p.add_argument('--max-workers', type=int, default=5)
    p.add_argument('--challenges', type=int, default=100)
    p.add_argument('--model', default='resnet20_quan')
    p.add_argument('--interval', type=float, default=0.5)
    p.add_argument('--logdir', default=None,
                   help='기본: ./bench_logs/<YYYYmmdd_HHMMSS>')
    return p.parse_args()


def nvidia_smi_used_mb():
    """GPU0 의 used VRAM (MB). 실패 시 빈 문자열."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().split('\n')
        return out[0].strip()
    except Exception:
        return ''


def gather_rss_gb(ps_objs):
    """psutil Process 리스트에 대한 (자식 포함) RSS 합 GB."""
    rss_per_proc = []
    for ps in ps_objs:
        try:
            r = ps.memory_info().rss
            for c in ps.children(recursive=True):
                try:
                    r += c.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            rss_per_proc.append(r / 1024**3)
        except psutil.NoSuchProcess:
            rss_per_proc.append(float('nan'))
    return rss_per_proc


def run_one(mode, n_workers, args, logdir):
    """단일 (mode, n_workers) 실행. 요약 dict 반환."""
    tag = f"{mode}_n{n_workers}"
    run_dir = logdir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "timeseries.csv"
    print(f"\n=== {tag} ===", flush=True)

    # 자식 프로세스용 환경변수
    env_base = os.environ.copy()
    if mode == 'cpu':
        env_base['CUDA_VISIBLE_DEVICES'] = ''
    # gpu 모드는 기본 CUDA_VISIBLE_DEVICES 그대로

    # 이전 sweep 잔여 데이터 청소 (워커별 디렉터리 비우기)
    for w in range(n_workers):
        d = SCRIPT_DIR / "data" / "challenge_bench" / f"{args.model}_w{w}" / "random"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    # 자식 프로세스 실행
    procs = []
    for w in range(n_workers):
        log_path = run_dir / f"worker_{w}.log"
        f = open(log_path, 'w')
        p = subprocess.Popen(
            [sys.executable, '-u', str(BENCH_SCRIPT),
             args.model, str(w), str(args.challenges)],
            cwd=SCRIPT_DIR, env=env_base,
            stdout=f, stderr=subprocess.STDOUT,
        )
        ps = psutil.Process(p.pid)
        procs.append({'w': w, 'popen': p, 'ps': ps, 'logfile': f})
        print(f"  spawned worker {w}  pid={p.pid}", flush=True)

    # 샘플링 루프
    t0 = time.time()
    headers = ['t'] + [f'w{i}_rss_gb' for i in range(n_workers)] \
              + ['total_rss_gb', 'vram_used_mb']
    peak_total = 0.0
    peak_vram = 0
    peak_per_worker = [0.0] * n_workers

    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(headers)
        while any(pr['popen'].poll() is None for pr in procs):
            t = time.time() - t0
            rss_list = gather_rss_gb([pr['ps'] for pr in procs])
            total = sum(r for r in rss_list if r == r)  # NaN 제외
            vram = nvidia_smi_used_mb()
            wr.writerow([f"{t:.2f}"] + [f"{r:.3f}" if r == r else ''
                                        for r in rss_list]
                        + [f"{total:.3f}", vram])
            f.flush()

            peak_total = max(peak_total, total)
            for i, r in enumerate(rss_list):
                if r == r:
                    peak_per_worker[i] = max(peak_per_worker[i], r)
            if vram.isdigit():
                peak_vram = max(peak_vram, int(vram))
            time.sleep(args.interval)

        # 종료 후 한 번 더 (자식 종료 직전 피크 캡처 보강)
        t = time.time() - t0
        # 자식이 이미 reaped 됐을 수 있어 skip

    # 정리
    for pr in procs:
        pr['popen'].wait()
        pr['logfile'].close()

    wall = time.time() - t0
    summary = {
        'mode': mode,
        'n_workers': n_workers,
        'wall_time_s': round(wall, 2),
        'peak_total_rss_gb': round(peak_total, 3),
        'peak_per_worker_avg_gb':
            round(sum(peak_per_worker) / n_workers, 3) if n_workers else 0,
        'peak_per_worker_max_gb':
            round(max(peak_per_worker), 3) if peak_per_worker else 0,
        'peak_vram_mb': peak_vram,
        'returncodes': [pr['popen'].returncode for pr in procs],
    }
    print(f"  → peak total RSS = {summary['peak_total_rss_gb']:.2f} GB  "
          f"peak VRAM = {summary['peak_vram_mb']} MB  "
          f"wall = {summary['wall_time_s']:.1f}s  "
          f"rc = {summary['returncodes']}",
          flush=True)
    # 실패한 워커 있으면 로그 위치 안내
    if any(rc not in (0, None) for rc in summary['returncodes']):
        print(f"  ⚠ non-zero return code. logs at {run_dir}/worker_*.log",
              flush=True)
    return summary


def main():
    args = parse_args()

    # 로그 디렉터리
    if args.logdir:
        logdir = Path(args.logdir)
    else:
        logdir = SCRIPT_DIR / "bench_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"logdir: {logdir}", flush=True)

    # GPU 사용 가능 여부 사전 체크 (gpu/both 모드일 때)
    if args.mode in ('gpu', 'both'):
        try:
            subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL, timeout=2)
        except Exception:
            print("⚠ nvidia-smi 호출 실패. GPU 모드 건너뜀.", flush=True)
            args.mode = 'cpu' if args.mode == 'both' else None
            if args.mode is None:
                sys.exit(1)

    modes = ['cpu', 'gpu'] if args.mode == 'both' else [args.mode]
    Ns = list(range(1, args.max_workers + 1))

    print(f"sweep plan: modes={modes}  N={Ns}  challenges={args.challenges}  "
          f"model={args.model}", flush=True)

    # master summary
    master_path = logdir / "summary.csv"
    fields = ['mode', 'n_workers', 'wall_time_s', 'peak_total_rss_gb',
              'peak_per_worker_avg_gb', 'peak_per_worker_max_gb',
              'peak_vram_mb', 'returncodes']

    rows = []
    for mode in modes:
        for n in Ns:
            row = run_one(mode, n, args, logdir)
            rows.append(row)
            # 즉시 master 갱신 (도중 중단 대비)
            with open(master_path, 'w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                wr.writeheader()
                for r in rows:
                    wr.writerow({k: r[k] for k in fields})

    # 최종 요약 출력
    print(f"\n{'='*60}\nDONE → {master_path}\n{'='*60}", flush=True)
    print(f"{'mode':<5} {'N':>2}  {'wall(s)':>8}  "
          f"{'peak_RSS(GB)':>13}  {'per_w_avg':>10}  "
          f"{'per_w_max':>10}  {'peak_VRAM(MB)':>14}")
    for r in rows:
        print(f"{r['mode']:<5} {r['n_workers']:>2}  "
              f"{r['wall_time_s']:>8.1f}  "
              f"{r['peak_total_rss_gb']:>13.2f}  "
              f"{r['peak_per_worker_avg_gb']:>10.2f}  "
              f"{r['peak_per_worker_max_gb']:>10.2f}  "
              f"{r['peak_vram_mb']:>14}")


if __name__ == "__main__":
    main()
