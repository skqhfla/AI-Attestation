#!/usr/bin/env python3
"""
generate_challenge_bench.py 를 모드(CPU/GPU) × 워커 수(N=1..5) 로 sweep 하면서
RAM(RSS) 과 VRAM 을 시계열로 로깅한다.

각 (mode, N) 조합마다:
  1) 깨끗한 출력 디렉터리에서 N 개 자식 프로세스 동시 실행
  2) interval 초마다 자식별 RSS + 합계 + nvidia-smi VRAM 샘플링
  3) per-run CSV + 마지막에 master summary CSV 저장

지원 모델 (generate_challenge_bench.py 와 동일):
  - resnet20_quan       (CIFAR-10,  32x32)
  - vgg11_quan          (CIFAR-10,  32x32)
  - wideresnet_quan     (CIFAR-10,  32x32)
  - efficientnetv2_quan (CIFAR-100, 224x224)
  - tresnet_quan        (CIFAR-100, 224x224)

사용 — 단일 sweep:
  # 동일 모델 N개 (스케일링 곡선)
  python sweep_resources.py --models resnet20_quan

  # 워커마다 다른 모델 (N=1..K 일 때 round-robin 배정)
  python sweep_resources.py --models resnet20_quan vgg11_quan wideresnet_quan

  # 큰 모델만 따로
  python sweep_resources.py --models efficientnetv2_quan --max-workers 3

사용 — 시나리오 일괄 실행 (조건별 자동화):
  # conditions.json 의 각 시나리오를 순차 실행 후 master summary 생성
  python sweep_resources.py --config conditions.json

옵션:
    --config FILE              JSON 시나리오 파일. 지정 시 아래 옵션은
                                 시나리오 미지정 필드의 기본값으로만 사용
    --mode both                (cpu | gpu | both, default: both)
    --max-workers 5            (default: 5; N=1..max_workers 까지)
    --challenges 100           (자식 1개당 생성할 이미지 수, default: 100)
    --models resnet20_quan ... (default: ['resnet20_quan']. 다수 지정 시
                                 워커별 round-robin 배정)
    --interval 0.5             (샘플링 주기 초, default: 0.5)
    --logdir bench_logs        (default: ./bench_logs/<timestamp>)

conditions.json 포맷:
    [
      {
        "name": "cifar10_solo",
        "mode": "both",
        "models": ["resnet20_quan"],
        "max_workers": 5,
        "challenges": 100
      },
      { ... 다음 시나리오 ... }
    ]
  (각 필드는 생략 가능. 생략 시 CLI 기본값 사용)
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_SCRIPT = SCRIPT_DIR / "generate_challenge_bench.py"

KNOWN_MODELS = [
    'resnet20_quan',
    'vgg11_quan',
    'wideresnet_quan',
    'efficientnetv2_quan',
    'tresnet_quan',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None,
                   help='JSON 시나리오 파일 경로. 지정 시 시나리오별 일괄 실행')
    p.add_argument('--mode', choices=['cpu', 'gpu', 'both'], default='both')
    p.add_argument('--max-workers', type=int, default=5)
    p.add_argument('--challenges', type=int, default=100)
    p.add_argument('--models', nargs='+', default=['resnet20_quan'],
                   choices=KNOWN_MODELS,
                   help='워커별 round-robin 으로 배정될 모델 목록')
    # 구버전 호환 — 단일 --model 도 받아준다
    p.add_argument('--model', default=None,
                   help='(deprecated) --models 의 단일값 alias')
    p.add_argument('--interval', type=float, default=0.5)
    p.add_argument('--logdir', default=None,
                   help='기본: ./bench_logs/<YYYYmmdd_HHMMSS>')
    args = p.parse_args()
    if args.model is not None:
        args.models = [args.model]
    return args


def load_conditions(path, cli_defaults):
    """JSON 시나리오 파일을 읽어 (name, SimpleNamespace) 리스트로 반환.
    각 필드가 시나리오에 없으면 cli_defaults 에서 채움."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("conditions JSON 은 list 여야 합니다.")

    out = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise ValueError(f"[{i}] 시나리오는 dict 여야 합니다.")
        name = c.get('name', f'scenario_{i}')
        mode = c.get('mode', cli_defaults.mode)
        if mode not in ('cpu', 'gpu', 'both'):
            raise ValueError(f"[{name}] mode 가 cpu/gpu/both 가 아님: {mode}")
        models = c.get('models', cli_defaults.models)
        for m in models:
            if m not in KNOWN_MODELS:
                raise ValueError(f"[{name}] 알 수 없는 모델: {m}")
        scen = SimpleNamespace(
            name=name,
            mode=mode,
            max_workers=int(c.get('max_workers', cli_defaults.max_workers)),
            challenges=int(c.get('challenges', cli_defaults.challenges)),
            models=models,
            interval=float(c.get('interval', cli_defaults.interval)),
        )
        out.append((name, scen))
    return out


def gpu_available():
    try:
        subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL, timeout=2)
        return True
    except Exception:
        return False


def run_scenario(name, scenario, base_logdir, gpu_ok):
    """단일 시나리오의 modes × Ns sweep 수행. 결과 rows(list of dict) 반환."""
    scen_logdir = base_logdir / name
    scen_logdir.mkdir(parents=True, exist_ok=True)

    # GPU 모드 가능 여부 fallback
    mode = scenario.mode
    if mode in ('gpu', 'both') and not gpu_ok:
        if mode == 'both':
            print(f"[{name}] ⚠ GPU 사용 불가 → cpu only 로 폴백", flush=True)
            mode = 'cpu'
        else:
            print(f"[{name}] ⚠ GPU 사용 불가 → 스킵", flush=True)
            return []
    modes = ['cpu', 'gpu'] if mode == 'both' else [mode]
    Ns = list(range(1, scenario.max_workers + 1))

    print(f"\n[{name}] plan: modes={modes}  N={Ns}  "
          f"challenges={scenario.challenges}  models={scenario.models}",
          flush=True)

    rows = []
    for m in modes:
        for n in Ns:
            row = run_one(m, n, scenario, scen_logdir)
            row['scenario'] = name
            rows.append(row)
    return rows


def assign_models(models, n_workers):
    """워커 인덱스 → 모델 이름. models 리스트를 round-robin 으로 배정."""
    return [models[w % len(models)] for w in range(n_workers)]


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


def read_worker_timings(worker_models):
    """bench 자식 프로세스가 남긴 _timing.json 들을 읽어 리스트로 반환.
    누락된 워커는 스킵."""
    out = []
    for w, m in enumerate(worker_models):
        path = SCRIPT_DIR / 'data' / 'challenge_bench' / f'{m}_w{w}' / '_timing.json'
        if path.exists():
            try:
                with open(path) as f:
                    out.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                pass
    return out


def run_one(mode, n_workers, args, logdir):
    """단일 (mode, n_workers) 실행. 요약 dict 반환."""
    tag = f"{mode}_n{n_workers}"
    run_dir = logdir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "timeseries.csv"
    worker_models = assign_models(args.models, n_workers)
    print(f"\n=== {tag}  models={worker_models} ===", flush=True)

    # 자식 프로세스용 환경변수
    env_base = os.environ.copy()
    if mode == 'cpu':
        env_base['CUDA_VISIBLE_DEVICES'] = ''
    # gpu 모드는 기본 CUDA_VISIBLE_DEVICES 그대로

    # 이전 sweep 잔여 데이터 청소 (워커별 디렉터리 비우기)
    for w, m in enumerate(worker_models):
        d = SCRIPT_DIR / "data" / "challenge_bench" / f"{m}_w{w}" / "random"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    # 자식 프로세스 실행
    procs = []
    for w, m in enumerate(worker_models):
        log_path = run_dir / f"worker_{w}_{m}.log"
        f = open(log_path, 'w')
        p = subprocess.Popen(
            [sys.executable, '-u', str(BENCH_SCRIPT),
             m, str(w), str(args.challenges)],
            cwd=SCRIPT_DIR, env=env_base,
            stdout=f, stderr=subprocess.STDOUT,
        )
        ps = psutil.Process(p.pid)
        procs.append({'w': w, 'model': m, 'popen': p, 'ps': ps, 'logfile': f})
        print(f"  spawned worker {w}  model={m}  pid={p.pid}", flush=True)

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

    # bench 자식이 남긴 워커별 timing JSON 집계
    timings = read_worker_timings(worker_models)
    gen_times = [t['gen_time_s'] for t in timings]
    pred_times = [t['pred_time_s'] for t in timings]
    total_times = [t['total_time_s'] for t in timings]

    def _avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    def _mx(xs):
        return round(max(xs), 3) if xs else 0.0

    summary = {
        'mode': mode,
        'n_workers': n_workers,
        'models': '|'.join(worker_models),
        'wall_time_s': round(wall, 2),
        'gen_time_avg_s': _avg(gen_times),
        'gen_time_max_s': _mx(gen_times),
        'pred_time_avg_s': _avg(pred_times),
        'pred_time_max_s': _mx(pred_times),
        'per_worker_total_avg_s': _avg(total_times),
        'per_worker_total_max_s': _mx(total_times),
        'peak_total_rss_gb': round(peak_total, 3),
        'peak_per_worker_avg_gb':
            round(sum(peak_per_worker) / n_workers, 3) if n_workers else 0,
        'peak_per_worker_max_gb':
            round(max(peak_per_worker), 3) if peak_per_worker else 0,
        'peak_vram_mb': peak_vram,
        'returncodes': [pr['popen'].returncode for pr in procs],
    }
    print(f"  → gen(max)={summary['gen_time_max_s']:.2f}s  "
          f"pred(max)={summary['pred_time_max_s']:.2f}s  "
          f"worker_total(max)={summary['per_worker_total_max_s']:.2f}s  "
          f"wall={summary['wall_time_s']:.1f}s",
          flush=True)
    print(f"     peak total RSS = {summary['peak_total_rss_gb']:.2f} GB  "
          f"peak VRAM = {summary['peak_vram_mb']} MB  "
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

    # 시나리오 빌드
    if args.config:
        scenarios = load_conditions(args.config, args)
        print(f"loaded {len(scenarios)} scenarios from {args.config}:", flush=True)
        for name, scen in scenarios:
            print(f"  - {name}: mode={scen.mode} N=1..{scen.max_workers} "
                  f"models={scen.models} challenges={scen.challenges}",
                  flush=True)
    else:
        scenarios = [('default', SimpleNamespace(
            name='default',
            mode=args.mode,
            max_workers=args.max_workers,
            challenges=args.challenges,
            models=args.models,
            interval=args.interval,
        ))]

    # GPU 사용 가능 여부 사전 체크 (한 번만)
    gpu_ok = gpu_available()
    if not gpu_ok:
        print("⚠ nvidia-smi 호출 실패 — GPU 시나리오는 폴백/스킵됨", flush=True)

    # master summary (시나리오별 column 포함)
    master_path = logdir / "summary.csv"
    fields = ['scenario', 'mode', 'n_workers', 'models', 'wall_time_s',
              'gen_time_avg_s', 'gen_time_max_s',
              'pred_time_avg_s', 'pred_time_max_s',
              'per_worker_total_avg_s', 'per_worker_total_max_s',
              'peak_total_rss_gb', 'peak_per_worker_avg_gb',
              'peak_per_worker_max_gb', 'peak_vram_mb', 'returncodes']

    all_rows = []
    for name, scen in scenarios:
        rows = run_scenario(name, scen, logdir, gpu_ok)
        all_rows.extend(rows)
        # 즉시 master 갱신 (도중 중단 대비)
        with open(master_path, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for r in all_rows:
                wr.writerow({k: r.get(k, '') for k in fields})

    # 최종 요약 출력
    print(f"\n{'='*110}\nDONE → {master_path}\n{'='*110}", flush=True)
    print(f"{'scenario':<24} {'mode':<4} {'N':>2}  "
          f"{'gen_max(s)':>10}  {'pred_max(s)':>11}  "
          f"{'wTotal_max':>10}  {'wall(s)':>7}  "
          f"{'peak_RSS(GB)':>12}  {'VRAM(MB)':>9}")
    last_scen = None
    for r in all_rows:
        scen_label = r['scenario'] if r['scenario'] != last_scen else ''
        last_scen = r['scenario']
        print(f"{scen_label:<24} {r['mode']:<4} {r['n_workers']:>2}  "
              f"{r['gen_time_max_s']:>10.2f}  "
              f"{r['pred_time_max_s']:>11.2f}  "
              f"{r['per_worker_total_max_s']:>10.2f}  "
              f"{r['wall_time_s']:>7.1f}  "
              f"{r['peak_total_rss_gb']:>12.2f}  "
              f"{r['peak_vram_mb']:>9}")


if __name__ == "__main__":
    main()
