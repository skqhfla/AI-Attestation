# AI-Attestation & BFA Framework (Backend)

이 프레임워크는 딥러닝 모델의 보안성을 평가하는 **Bit-Flip Attack(BFA)**과 모델의 무결성을 검증하는 **AI-Attestation** 기능을 통합한 백엔드 시스템입니다. 엣지 디바이스(Edge AI) 환경에서 모델이 위변조되지 않았음을 증명하기 위한 전체 파이프라인을 제공합니다.

---

## 🚀 주요 기능 (Key Features)

### 1. Bit-Flip Attack (BFA) & Defense
  - **Attack Engine**: 양자화된 신경망(DNN)에서 성능을 급격히 저하시키는 취약한 비트를 검색하고 공격합니다.
  - **Defense Strategy**: Piecewise Weight Clustering 및 Binarization-aware training을 통해 공격에 강인한 모델을 학습합니다.

### 2. AI-Attestation (Model Integrity Verification)
  - **Challenge Generation**: 모델의 특정 출력 분포를 유도하는 최적화된 노이즈 이미지를 생성합니다.
  - **Verification Engine**: 엣지 디바이스에서 반환된 응답(Logits)과 사전에 정의된 지문(Golden Fingerprint)을 비교하여 위변조 여부를 판별합니다.

---

## 📂 디렉토리 구조 및 역할 (Directory Structure)

```text
AI-Attestation/
├── attestation/             # AI 모델 증명 핵심 로직
│   ├── generate_challenge.py # 일반화된 챌린지 이미지 생성 엔진
│   ├── attack.py             # 증명 시스템에 대한 공격 시뮬레이션
│   └── train_model.py        # 증명용 모델 학습 로딩
├── attack/                  # BFA 공격 관련 소스 코드
├── models/                  # 양자화 및 다양한 신경망 아키텍처
│   ├── quan_resnet_cifar.py  # ResNet 계열 양자화 모델
│   ├── quan_wyze.py         # Wyze 모델 특화 양자화 아키텍처
│   └── quantization.py      # STE(Straight-Through Estimator) 등 양자화 모듈
├── data/                    # 모델 가중치 및 생성된 챌린지 저장
├── test_challenge.py        # (Main) 생성된 챌린지 기반 검증 테스트
├── BFA.py                   # (Core) Bit-Flip Attack 엔진 실행 파일
└── main.py                  # 모델 학습 및 기본 평가 진입점
```

---

## 🛠 핵심 프로세스 흐름 (Workflow)

### 1단계: 모델 준비 및 취약성 분석
- 모델을 8-bit 등으로 양자화하여 준비합니다. (`models/`)
- BFA를 통해 모델의 어떤 레이어나 비트가 공격에 취약한지 분석합니다. (`BFA.py`)

### 2단계: 챌린지 생성 (Generate Challenge)
- `generate_challenge_wyze.py` 또는 `attestation/generate_challenge.py`를 통해 챌린지 이미지를 생성합니다.
- 이 이미지는 모델에 입력되었을 때 특정 클래스의 확률을 균등하게(예: 20%씩) 만드는 등의 특수 목적을 가집니다.

### 3단계: 무결성 검증 (Verification)
- 챌린지 이미지를 대상 디바이스로 전송합니다.
- 디바이스의 출력값과 기준값을 `test_challenge.py` 로직을 통해 비교하여 모델 위변조를 판별합니다.

---

## 💻 환경 설정 (Environment Setup)

- **Python**: 3.6+ (Anaconda 권장)
- **Framework**: PyTorch >= 1.0.1
- **Dependencies**: 
  ```bash
  pip install torch torchvision numpy Pillow timm
  ```

---

## 📝 사용 예시 (Usage)

### 챌린지 이미지 생성
```bash
python generate_challenge.py
```

### 모델 검증 테스트
```bash
python test_challenge.py
```

