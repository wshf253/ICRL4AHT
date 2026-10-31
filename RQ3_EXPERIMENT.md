# RQ3: 상대 정보가 작업 선택과 협업 성능에 영향을 주는가?

이 구현은 Continual ToM 전체 실험에 앞선 **상대 정보 개입 실험**이다.
같은 초기 상태에서 `none / correct / opposite` 정보만 바꾼다.
상대 메모리를 경험으로 학습하는 Day 1–5 루프는 아직 포함하지 않는다.
우선 정확한 상대 정보를 제공해도 계획과 성능에 차이가 없다면, 메모리 학습보다
시나리오/계획기의 정보 활용 능력을 먼저 검토한다.

## 구현된 시나리오

- OvercookedV2 기반의 작은 두 냄비 주방. 원본 benchmark layout/평가를 수정하지 않는다.
- 공개된 고정 레시피: 재료 0 세 개. 맵과 자신의 위치도 공개 정보다.
- 시작할 때 한 냄비는 조리 중이고, 다른 냄비는 재료가 부족하다.
- seed로 조리 중인 냄비 위치, 남은 조리 시간, 다른 냄비의 재료 수를 결정한다.
- 각 seed의 모든 상대/정보 조건은 정확히 같은 초기 상태에서 출발한다.
- 상대는 `prep` 또는 `delivery` 우선순위만 다르고, 이동/상호작용 능력은 동일하다.
  선호 작업이 없으면 다른 일을 한다. 완료/목표 변경/timeout까지 작업에 전념한다.
- 두 agent 모두 같은 결정론적 실행기를 사용한다. LLM은 작업을 선택하고, 실행기는
  BFS 경로 탐색, 재료/접시 획득, 냄비 투입, 수프 수거와 배달을 수행한다.
- 두 실행기가 같은 결정론적 규칙을 쓰면 대칭 상황(같은 빈 칸에 동시 진입, 서로의 유일한
  접근 칸을 막음, 거울 대칭 우회)에서 같은 충돌이 끝없이 반복된다. 이를 막기 위해 충돌·막힘·
  A-B-A 왕복이 관측되면 1/2 확률로 양보하는 무작위 backoff를 쓴다. 난수는 (trial seed,
  agent id)로 시드되어 재현 가능하며 profile/조건과 무관하다. 이 수정 이전 결과는 이동
  livelock이 섞여 있으므로 조건 비교에 사용하지 않는다.
- LLM을 매 이동 step 호출하지 않고 작업이 끝나거나 무효화되면 호출한다.
- 기본 `--steps 200`, `--max-job-steps 32`. 고정된 환경 step 예산이며 wall-clock 예산은 아니다.

이 partner는 기존 utility-greedy의 버그를 고친 버전이 아니라, 연구 가설을 통제해서
검증하기 위한 별도 정책이다. 기존 heuristic과 평가 pipeline은 보존한다.

## 비교 조건

| 조건 | planner에 제공되는 상대 정보 |
|---|---|
| none | 사전 정보 없음 |
| correct | 실제 partner의 작업 우선순위 설명 |
| opposite | 반대 partner의 작업 우선순위 설명 |

기본 실행은 3 seeds × 2 partner profiles × 3 conditions = **18 trials**.
LLM은 실제 profile ID, condition 이름, 상대 내부 job을 받지 않는다.
현재 관측, 후보 작업, 최근 자기 작업 이력, 해당 조건의 상대 설명만 받는다.
correct/opposite의 설명은 같은 문장 틀이고, 동일한 생성/문맥 예산을 사용한다.
none은 정보가 적으므로 실제 입력 토큰 수가 동일하지는 않다. 사용량은 LLM 응답에 기록된다.

## 설치 및 서버 환경 재사용

```bash
cd ~/work/icrl4aht
git pull --ff-only
uv sync --locked
```

기존 PARTNR `scripts/run_qwen35_partnr.sh`에 있는 설정을 재사용한다:

- serving venv: `~/work/llm-serving/.venv-qwen35`
- 모델 경로: `/lab/ckpt/jsj412/Qwen3.5-27B`
- API model ID: `qwen35-27b`
- bf16, 1 GPU, text-only, 16K context, max-num-seqs 1

이는 GitHub 코드에서 확인한 경로다. 실제 서버 디렉터리의 존재나 GPU 실행을
이 작업 환경에서 확인한 것은 아니다. `llm_serving`처럼 경로가 다르면 아래 변수를 지정한다.

```bash
export LLM_SERVE_ENV="$HOME/work/llm_serving/.venv-qwen35"
# 모델 위치가 다를 때만 지정
export LLM_MODEL_PATH=/path/to/Qwen3.5-27B
```

ICRL의 가상환경에는 vLLM/torch를 설치하지 않는다. 서빙 venv와 모델 파일을 재사용하고,
ICRL은 CPU JAX + HTTP client로 실행한다. Qwen3.5는 기본적으로 thinking이 켜질 수 있어
요청에 `chat_template_kwargs: {enable_thinking: false}`를 넣는다. 모델 응답은 JSON으로
검증하며 잘못된 응답을 heuristic 행동으로 조용히 대체하지 않는다.

## 첫 실제 모델 실험 — 한 job에서 서버와 실험 함께 실행

```bash
lab-run --new --gpus 1 -- bash scripts/run_qwen35_rq3.sh --video
```

이 명령은 기존 서빙 환경으로 모델을 한 번 올린 뒤 18 trials를 순차 실행한다.
새 server는 해당 job 안에서만 사용하고, 실험 종료 시 그 server만 정리한다.
결과 경로는 `outputs/rq3_<시간>_<job>_<pid>/experiment/`이다.
18개 영상이 필요 없다면 `--video`를 빼면 된다.

기존 서버를 이미 띄워둔 경우 직접 호출할 수 있다:

```bash
export ICRL_LLM_BASE_URL=http://MODEL_NODE:PORT/v1
export ICRL_LLM_MODEL=qwen35-27b
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu \
  uv run --locked python -m research.coordination.run \
  --base-url "$ICRL_LLM_BASE_URL" --model "$ICRL_LLM_MODEL" \
  --out outputs/rq3_existing_server
```

`MODEL_NODE:PORT`는 실제 접근 가능한 주소로 교체한다. `127.0.0.1`은 **실험 프로세스와
모델 서버가 같은 노드일 때만** 통한다. 기존 PARTNR launcher는 loopback에 바인딩하며
작업 종료 시 서버도 종료하므로, 그 폴더 재사용과 실행 중인 endpoint 재사용은 다르다.
다른 노드에서 접근할 수 없는 loopback server라면 첫 번째 같은-job 명령이 간단하다.
API 인증이 설정된 경우 `ICRL_LLM_API_KEY`를 환경변수로 지정한다. 키는 로그에 저장하지 않는다.

## 실행 옵션

```bash
# 더 많은 paired seeds. 실험 전 seed 목록을 고정한다.
lab-run --new --gpus 1 -- bash scripts/run_qwen35_rq3.sh --seeds 0 1 2 3 4 --steps 200

# 부분 관측 확장 (기본 RQ3 full 결과와 섞지 않는다)
lab-run --new --gpus 1 -- bash scripts/run_qwen35_rq3.sh --observation partial --video

# 모델 없이 환경/파일 저장만 검증. 연구 성능 결과로 사용 금지.
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu \
  uv run --locked python -m research.coordination.run \
  --backend scripted --seeds 0 --out outputs/rq3_scripted
```

`--backend scripted`는 상대 설명을 의도적으로 무시한다. 따라서 같은 profile 내 세
조건의 결과가 같아야 한다. 이는 동작 검사용 음성 대조이며, ToM 효과를 만들어내지 않는다.

기본 full 관측은 RQ3의 인과 비교에서 관측/탐색 실패를 분리하기 위한 선택이다.
partial에서는 두 agent 모두 반경 2의 **실제 env observation**을 해석한다.
전체 state나 partner 내부 정보는 실행기에 전달하지 않는다. 공개 맵/자기 위치를
사용한 BFS는 허용하되, 보이지 않는 냄비 내용이나 상대 소지품은 unknown이다.
이 설정은 공개 맵과 고정 recipe를 갖는 별도 통제 과제이며 원본 benchmark 점수와
직접 비교하면 안 된다.

## 출력과 해석

각 trial 폴더:

- `summary.json`: 상태, seed/profile/condition, 초기 상태 hash, 보상/배달수/지표
- `steps.jsonl`: 평가자용 실제 행동과 두 agent 작업, event, 위치
- `decisions.jsonl`: planner 입력, 예측, 선택 작업, 관측 이후 평가용 실제 상대 작업
- `llm.jsonl`: 실제 request/response, 사용 토큰, 지연시간, 재시도/오류
- `rollout.mp4`: `--video` 사용 시 전체 상태 진단 영상. LLM 입력이 아니다.
- `error.log`: 오류가 난 trial에서 traceback

상위 폴더:

- `config.json`: 설정, commit, 연구 모듈 source hash, 관측/개입 범위
- `results.json`: 성공·실패 trial 전체 및 seed/profile별 paired 차이

주요 분석:

1. `correct_minus_none`: 상대 정보 제공의 성능 차이
2. `correct_minus_opposite`: 정보 내용의 상대 적합성이 만드는 차이
3. `first_choice_changed_correct_vs_opposite`: 초기 관측이 같은 상태에서 선택이 바뀌는가
4. `choice_opportunities`: 재료/배달 두 종류를 선택할 수 있었던 결정 횟수
5. `duplicate_demand_steps`: 같은 수프를 동시에 배달하려 하거나, 재료 하나만 부족한
   같은 냄비에 둘 다 재료 준비 작업을 맡은 step. 충돌 가능성 지표이며 실제 낭비의 완전한 측정은 아니다.
6. `agent_0/1_correct_delivery`, `ingredient_added`: 기여가 한쪽에만 몰리는지 확인
7. `agent_0/1_failed_move_steps`: 빈 바닥으로 이동했지만 위치가 바뀌지 않은 step
   (agent 충돌/막힘). 수십 이상이면 작업 선택이 아니라 이동 단계가 성능을 결정한 trial이다.
8. `partner_prediction_accuracy`: 상대가 현재 수행 중인 작업 종류 예측. 의도/선호 이해의
   단독 증거가 아니며 반드시 실제 성능과 선택 변화를 함께 본다.

실패 trial은 성능 0으로 취급하지 않고 status=error로 보존한다. 누락/실패한 조건이 있는
pair는 paired 효과에서 제외하고 `excluded_pairs`에 명시한다. 어떤 seed든 잘 나온 것만
선택해서 보고하지 않는다. 요약 평균만으로 유의성이나 인과 메커니즘 전체를 주장하지 않는다.

같은 상태에서 정보만 바꾼 초기 선택 비교는 직접적인 개입이다. 그 이후 rollout은 행동에
따라 상태가 달라지므로, 후반 선택을 같은 상태에서의 counterfactual로 간주하지 않는다.

## 검증과 범위

```bash
JAX_PLATFORMS=cpu uv run --locked python -m pytest -q tests/research/test_coordination.py
```

paired 초기 상태, 초기 선택 가능성, 부분 관측 정보 누출 방지, 실제 요리 실행,
scripted 조건 동일성, HTTP 모델 연동 및 오류 보존을 검사한다.
테스트의 HTTP endpoint는 가짜 응답 서버이며 Qwen 성능 검증이 아니다.
실제 Qwen3.5-27B 실행은 위 서버 명령으로 수행해야 한다.

다음 단계는 결과에서 정보→선택→성능의 연결을 확인한 뒤, correct 조건의 외부 설명을
반복 협업 경험에서 갱신한 상대 메모리로 교체하는 것이다. 그때 안정/일시적 제약/지속적
변화 조건을 별도로 추가한다.
