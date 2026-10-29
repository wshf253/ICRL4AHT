# ICRL4AHT 연구용 설치와 teammate 미리보기

앞으로 실험은 이 fork를 기준으로 진행한다. 환경 구현은 `envs/overcooked_v2/`,
기존 상대 정책은 `agents/overcooked_v2/`에 포함되어 있다. 앞서 만든 별도 JaxMARL
checkout은 필요하지 않다. 패키지 의존성인 `jaxmarl==0.0.7`은 uv가 설치한다.
기존 환경·정책·평가 로직은 변경하지 않았다.

## 1. 설치 (iiscluster, CPU부터 시작)

```bash
mkdir -p ~/work
git clone https://github.com/wshf253/ICRL4AHT.git ~/work/icrl4aht
cd ~/work/icrl4aht
uv sync --locked
```

이미 clone했다면 그 폴더에서 `git pull --ff-only` 후 `uv sync --locked`.
작업 중인 변경 사항은 먼저 확인한다. `.python-version`은 Python 3.11,
`uv.lock`은 의존성 버전을 고정한다. PARTNR/JaxMARL의 가상환경을 재사용하지 않는다.

원본의 JAX 0.5.3 및 학습 라이브러리 버전을 유지하되 기본 JAX 설치를 CPU로 바꿨다.
CUDA 12 설치는 `uv sync --locked --extra cuda12`로 선택 가능하지만 여기서는
CPU rollout만 검증했다. 특히 Pro6000 Blackwell에서 이 오래된 JAX/CUDA 조합의
GPU 지원은 별도 확인이 필요하므로 우선 CPU로 실행한다.

## 2. 실제 heuristic 행동 및 MP4 확인

저장소 루트에서 실행:

```bash
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu uv run --locked python -m scripts.preview_teammates --out outputs/preview_001
```

일반 PC에서는 `lab-run --new --gpus 0 --` 부분을 빼면 된다.
기본값: `grounded_coord_ring`, agent 0 = utility_greedy,
agent 1 = territory, seed 0, 400 steps. 학습이나 체크포인트 다운로드는 없다.
첫 실행에는 JAX 컴파일과 정책 경로 캐시 생성이 필요하다.

생성 파일:

- `outputs/preview_001/rollout.mp4`: 전체 상태를 보여주는 진단 영상, 8 FPS
- `outputs/preview_001/rollout.jsonl`: step별 행동·보상·위치·소지품·종료 여부
- `outputs/preview_001/summary.json`: 설정·코드 버전·JAX 장치·누적 보상·배달 지표

기존 결과가 있는 출력 폴더는 덮어쓰지 않는다. 재실행은 새로운 `--out`을 지정한다.
영상은 Pillow로 그린 진단용 격자이며, 논문의 원본 렌더러나 agent 입력 영상은 아니다.
`P`는 접시, `C`는 조리 완료, 숫자는 재료 ID, `t`는 timer/indicator 값이다.
상단의 shared return은 두 agent에게 동일하게 주어지는 보상이다.
`returns_per_agent`를 합쳐 팀 보상을 두 배로 계산하지 않는다.
`steps_with_correct_delivery`는 배달이 발생한 timestep 수이며, 동시에 배달할 수
있으므로 일반적으로 정확한 배달 음식 수와 동일하다고 간주하지 않는다.

## 3. 상대 조합 변경

```bash
# 재료 담당 + 배달 담당 조합 (맵과 역할에 따라 막히거나 보상 0일 수 있음)
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu uv run --locked python -m scripts.preview_teammates --layout cramped_room --agent0 assembly_line --agent1 assembly_line --role0 runner --role1 plater --out outputs/roles_001

# recipe-aware 정책도 실행 가능
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu uv run --locked python -m scripts.preview_teammates --agent0 recipe_aware_button --agent1 territory --out outputs/recipe_001

# 영상 없이 로그만 저장
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu uv run --locked python -m scripts.preview_teammates --video none --seed 1 --out outputs/no_video_001
```

지원 family: `assembly_line`, `territory`, `utility_greedy`, `recipe_aware_button`.
`--role0/--role1`의 runner/plater 설정은 assembly_line에만 적용한다.
이 미리보기는 각 family의 기본 파라미터를 사용하며, 논문의 test manifest에
정의된 전체 상대 분포를 평가하는 프로그램이 아니다.

## 4. 관측과 평가 범위

`make_env`의 benchmark 설정을 사용한다: 시야 반경 2, agent 위치 랜덤 초기화,
배달 후 recipe 재표본화, 잘못된 배달에 대한 penalty 등. `--steps` 기본값은 400이다.
종료 직후 자동 reset 화면이 영상에 들어가지 않도록 factory가 생성한 환경의
`step_env`를 호출하고 done에서 멈춘다. 기존 평가 runner는 수정하지 않는다.

**환경 obs가 부분 관측이어도, 이 미리보기의 기존 heuristic은 전체 state를 읽는다.**
두 agent 모두 privileged baseline이며, 부분 관측 기반 LLM 성능으로 해석하면 안 된다.
영상·위치 로그·recipe 정보도 평가자용이다. 향후 LLM 입력에는 agent의 obs와
허용된 이력만 전달하도록 별도 adapter를 만들고, 상대 policy의 정보 접근 범위도
실험 설정에 명시한다.

`status=completed`는 실행 종료를 의미하며, 성공적인 협업을 보장하지 않는다.
실제로 cramped_room의 assembly_line 기본 조합에서는 보상 0을 관측했다.
기본 mixed 조합은 seed 0 / 400 steps에서 shared return 180,
배달 발생 9 timesteps를 관측했다. 이는 설치 검증용 단일 rollout이며 성능 결론이 아니다.

## 5. 이후 연구 코드 구성

- `scripts/preview_teammates.py`: 설치·상대 행동 확인
- 이후 LLM adapter, 상대 메모리, 반복 episode runner는 별도 모듈로 추가
- 원본 `eval_icrl.py`와 test manifest는 기존 비교 기준으로 유지
- `outputs/`, `checkpoints/`, `runtime_memory/`, `.env` 등은 Git 제외
- 초기 프롬프트·메모리 템플릿·실험 설정은 Git으로 추적

현재 추가된 범위에는 LLM 호출, 지속 메모리 업데이트, AD/DPT 재학습이 포함되지 않는다.
