# Conference Presentation Script

English primary, Korean translation indented below each paragraph.
Slide numbers match the current deck (12 slides).

---

## Slide 1 — Title / One-line contribution

Hello, I'm presenting our work on a physically closed-loop HILS system using a Stewart platform for vehicle attitude-control validation.

> 안녕하세요, 스튜어트 플랫폼을 이용한 물리적 폐루프 HILS 기반 비행체 자세제어 검증 시스템에 관한 연구를 발표합니다.

In one sentence, our contribution is this: we connect X-Plane, a real Stewart platform, and a real PX4 flight controller so that simulated dynamics drive physical motion, and measured physical attitude feeds back into the simulator loop.

> 한 문장으로 요약하면, X-Plane과 실제 스튜어트 플랫폼, 실제 PX4 비행 제어기를 연결하여 시뮬레이션 동역학이 물리적 모션을 구동하고, 측정된 물리적 자세가 시뮬레이터 루프로 피드백되는 시스템을 구축했습니다.

Today I'll cover motivation, architecture, experiment design, quantitative results, limitations, and next steps.

> 오늘은 연구 동기, 아키텍처, 실험 설계, 정량적 결과, 한계, 향후 과제를 다루겠습니다.

`>> NEXT SLIDE`

## Slide 2 — Why add physical motion to HILS?

Conventional HILS is powerful, but in many setups the flight controller receives synthesized sensor signals, and attitude is checked mainly through screens and logs.

> 기존 HILS는 강력하지만, 많은 구성에서 비행 제어기가 합성 센서 신호를 수신하고 자세는 화면과 로그로만 확인합니다.

Our motivation is to add physical realism: the FC's IMU should experience real inertial excitation, not only virtual values.

> 우리의 동기는 물리적 사실성을 추가하는 것입니다. FC의 IMU가 가상 값만이 아니라 실제 관성 자극을 경험해야 합니다.

So this work asks: Can we build a stable, repeatable, physically closed-loop testbed, and quantify tracking error and latency?

> 본 연구의 질문: 안정적이고 반복 가능한 물리적 폐루프 시험 환경을 구축하고, 추종 오차와 지연을 정량화할 수 있는가?

`>> NEXT SLIDE`

## Slide 3 — What is new here?

The key differentiation is not just hardware presence, but physical sensing in the loop.

> 핵심 차별점은 단순히 하드웨어의 존재가 아니라, 루프 내 물리적 센싱입니다.

In our system, simulated attitude is realized by platform motion, and the PX4 estimator responds to that real motion. This creates a more realistic closed-loop behavior within platform limits.

> 본 시스템에서는 시뮬레이션 자세가 플랫폼 모션으로 구현되고, PX4 추정기가 실제 모션에 반응합니다. 이것이 플랫폼 한계 내에서 보다 현실적인 폐루프 거동을 만듭니다.

At the same time, I want to be explicit: this does not replace conventional HILS. Workspace limits and missing aerodynamic force reproduction mean it is a complementary validation tool.

> 동시에 명확히 하고 싶습니다: 이것은 기존 HILS를 대체하지 않습니다. 작업 영역 한계와 공력 미재현은 이것이 보완적 검증 도구임을 의미합니다.

`>> NEXT SLIDE`

## Slide 4 — System in action

[Show hardware photo and/or demo video]

Here you can see the actual system operating. The Stewart platform physically moves to match the simulated aircraft attitude, and the PX4 mounted on top senses that motion in real time.

> 여기서 실제 시스템 구동 모습을 보실 수 있습니다. 스튜어트 플랫폼이 시뮬레이션 항공기 자세에 맞게 물리적으로 구동되고, 위에 탑재된 PX4가 실시간으로 해당 모션을 감지합니다.

`>> NEXT SLIDE`

## Slide 5 — How the physical closed loop works

Here is the full dataflow.

> 전체 데이터 흐름입니다.

[Show Fig. 1: System architecture/data flow with numbered callouts]

Step-by-step:
1) X-Plane publishes vehicle state to the Python host over UDP.
2) Host converts state to pose commands for the Stewart platform.
3) PX4 physically senses platform motion through its onboard IMU.
4) PX4 attitude returns to the host; host injects control back into X-Plane.

> 단계별로:
> 1) X-Plane이 UDP를 통해 비행체 상태를 Python 호스트로 전송.
> 2) 호스트가 상태를 스튜어트 플랫폼 pose 명령으로 변환.
> 3) PX4가 탑재 IMU를 통해 플랫폼의 물리적 모션을 감지.
> 4) PX4 자세가 호스트로 귀환하여 X-Plane에 제어 입력 주입.

So the loop is fully closed across software, mechanics, sensing, and control I/O.

> 소프트웨어, 기구부, 센싱, 제어 I/O를 아우르는 완전한 폐루프 구성입니다.

`>> NEXT SLIDE`

## Slide 6 — What we tested and how we measured it

The platform's neutral position is set at z=+20 mm to maximize the available rotation range.

> 플랫폼의 기본 자세는 회전 범위 극대화를 위해 z=+20 mm으로 설정되었습니다.

Our baseline scenario is a trim transition: pitch setpoint goes from 0 degrees to +5 degrees, holds, then returns.

> 기본 시나리오는 트림 전환입니다: 피치 설정값이 0도에서 +5도로 이동, 유지, 복귀.

Each run starts from the same simulator initial condition and is repeated multiple times for statistics.

> 각 실험은 동일한 시뮬레이터 초기 조건에서 시작하며 통계를 위해 반복 수행합니다.

Outputs per run include X-Plane attitude, PX4 attitude, command/ACK timing, and post-processed metrics. This baseline was selected because it is simple, repeatable, and directly supports quantitative evaluation.

> 실험별 출력은 X-Plane 자세, PX4 자세, 명령/ACK 타이밍, 후처리 메트릭을 포함합니다. 이 기본 시나리오는 단순하고, 반복 가능하며, 정량 평가를 직접 지원하므로 선택되었습니다.

`>> NEXT SLIDE`

## Slide 7 — Result 1: Closed-loop tracking is achieved

[Show Fig. 2: pitch response overlay]

In the overlay, PX4 pitch follows the commanded transition with expected lag. Raw error includes a near-constant offset component.

> 오버레이에서 PX4 피치가 예상된 지연으로 명령 전환을 추종합니다. 원시 오차에는 거의 일정한 오프셋 성분이 포함되어 있습니다.

That offset is not random instability. Evidence suggests a mounting/frame bias component, so we also evaluate bias-corrected error.

> 이 오프셋은 무작위 불안정이 아닙니다. 장착/프레임 바이어스 성분으로 보이므로, 바이어스 보정 오차도 평가합니다.

Main point: closed-loop tracking is achieved and quantifiable, with error structure that is explainable.

> 핵심: 폐루프 추종이 달성되었으며 정량화 가능하고, 오차 구조가 설명 가능합니다.

`>> NEXT SLIDE`

## Slide 8 — Result 2: Residual error is interpretable

[Show Fig. 3: pitch error + Fig. 4: mounting bias evidence]

A substantial portion of the observed pitch error behaves like a mounting bias. Correcting for it changes the interpretation of fidelity significantly.

> 관측된 피치 오차의 상당 부분이 장착 바이어스처럼 거동합니다. 이를 보정하면 충실도 해석이 크게 달라집니다.

This is important scientifically: the dominant residual error is not random instability. A fixed bias component must be separated from dynamic tracking fidelity.

> 이것은 과학적으로 중요합니다: 주요 잔여 오차는 무작위 불안정이 아닙니다. 고정 바이어스 성분을 동적 추종 충실도와 분리해야 합니다.

`>> NEXT SLIDE`

## Slide 9 — Result 3: Latency sets the fidelity limit

[Show Fig. 5(a): latency time series + Fig. 5(b): latency distribution]

We break timing into sample age, serial RTT, and end-to-end delays. Key numbers: serial RTT about 22 ms, PX4 rx-to-ACK about 41 ms, X-Plane rx-to-ACK about 51 ms.

> 타이밍을 sample age, serial RTT, 종단간 지연으로 분해합니다. 주요 수치: serial RTT 약 22 ms, PX4 rx→ACK 약 41 ms, X-Plane rx→ACK 약 51 ms.

Latency is not a side note; it directly explains observed phase lag and bounds achievable closed-loop fidelity.

> 지연은 부차적인 내용이 아닙니다. 관측된 위상 지연을 직접 설명하며 달성 가능한 폐루프 충실도를 한정합니다.

In baseline runs, saturation was minimal or absent, so these results primarily represent nominal-region behavior.

> 기본 실험에서 포화는 거의 발생하지 않았으므로, 이 결과는 주로 정상 영역 거동을 나타냅니다.

`>> NEXT SLIDE`

## Slide 10 — What this platform does not claim

I want to clearly state limitations:

> 한계를 명확히 하겠습니다:

- Finite workspace: large multi-axis commands reduce the feasible operating region.
- No physical aerodynamic load: motion cues are reproduced, not full external-force physics.
- Platform-level actuation limits: timing, hysteresis, and actuator constraints shape response fidelity.
- Not a flight replacement: this complements conventional HILS and real flight testing.

> - 유한한 작업 영역: 대각도 다축 명령 시 구동 가능 영역 축소.
> - 공력 하중 미재현: 모션 큐 재현, 외력 물리 미구현.
> - 플랫폼 구동 한계: 타이밍, 히스테리시스, 액추에이터 제약이 응답 충실도에 영향.
> - 비행 대체 불가: 기존 HILS 및 실제 비행 시험을 보완.

The correct claim is: a practical, measurable, physically informed pre-flight validation layer.

> 올바른 주장: 실용적이고 측정 가능한 물리 기반 사전 비행 검증 계층.

`>> NEXT SLIDE`

## Slide 11 — Takeaways

Three takeaways:

> 세 가지 핵심 성과:

First, integration: physically closed-loop HILS is successfully implemented with X-Plane, Stewart platform, and PX4.

> 첫째, 통합: X-Plane, 스튜어트 플랫폼, PX4를 연동한 물리적 폐루프 HILS를 성공적으로 구현.

Second, fidelity: closed-loop tracking is achieved and can be evaluated quantitatively.

> 둘째, 충실도: 폐루프 자세 추종이 달성되었으며 정량적으로 평가 가능.

Third, transparency: bias and latency are measured and used to define the platform's true fidelity, not hidden.

> 셋째, 투명성: 바이어스와 지연을 측정하여 플랫폼의 실제 충실도를 정의, 은폐하지 않음.

Near-term next steps: time-aligned error analysis, repeatability/hysteresis characterization, stress tests near saturation boundaries.

> 향후 과제: 시간 정렬 기반 오차 분석, 반복성/히스테리시스 특성 분석, 포화 한계 근처 스트레스 시험.

Thank you. I welcome questions.

> 감사합니다. 질문 환영합니다.

`>> BACKUP SLIDE (if needed)`

## Slide 12 — Q&A backup

Use this slide only if a specific question comes up. Otherwise stay on Slide 11.

> 특정 질문이 나올 때만 이 슬라이드를 사용. 그 외에는 슬라이드 11에서 대기.

---

## Optional live delivery cues

- If time is short, compress Slides 5 and 6 into one minute total.
- Spend most technical depth on Slides 5, 7, and 9.
- Slides 4 (demo footage) is for audience engagement — keep it brief, 15–20 seconds.

---

## Q&A prepared answers

**"Why z=+20 mm as the neutral position?"** → Slide 12, left panel with proof figure.
The rotation range is a function of z height. At z=+20 mm we get ~20° of rotation; at z=0 mm only ~10°. This maximizes the usable attitude envelope.

> 회전 범위는 z 높이의 함수입니다. z=+20 mm에서 ~20° 회전 범위, z=0 mm에서 ~10°만 확보. 자세 구현 범위를 극대화하는 선택.

**"Why not just conventional HILS?"** → Slides 3 + 10.
We add real inertial excitation that sensor synthesis cannot provide, while honestly acknowledging workspace and aerodynamic reproduction limits.

> 센서 합성이 제공할 수 없는 실제 관성 자극을 추가하면서, 작업 영역과 공력 재현 한계를 솔직히 인정.

**"How do you handle saturation?"** → Slides 6 + 10 + 11.
Baseline runs operate at ±5° out of ~20° available. Saturation was minimal. Stress tests near saturation boundaries are a planned next step.

> 기본 실험은 ~20° 범위 중 ±5°에서 운용. 포화 최소. 포화 한계 근처 스트레스 시험은 향후 과제.

**"What about yaw?"** → Slide 6.
Yaw is in the loop, but the trim-transition scenario primarily excites pitch. Yaw tracking and bias alignment are handled in the code but not the focus of this presentation.

> Yaw는 루프에 포함되나, 트림 전환 시나리오는 주로 피치를 여기. Yaw 추종 및 바이어스 정렬은 코드에서 처리하나 본 발표의 초점은 아님.

**"What's the effective bandwidth of this system?"** → Slides 5 + 9.
The host ticks at ~50 Hz. End-to-end latency from X-Plane receive to Stewart ACK is ~51 ms. This means the system is suited for low-frequency maneuvers like trim transitions; high-frequency dynamic content is filtered by the latency chain. Increasing bandwidth would require reducing serial overhead or moving to a faster transport.

> 호스트 틱 ~50 Hz. X-Plane 수신에서 Stewart ACK까지 종단간 지연 ~51 ms. 트림 전환 같은 저주파 기동에 적합하며, 고주파 동역학은 지연 체인에 의해 필터링됨. 대역폭 향상을 위해서는 시리얼 오버헤드 감소 또는 더 빠른 전송 수단이 필요.

**"The platform is open-loop — how do you trust pose accuracy?"** → Slide 10.
We command inverse kinematics but have no encoder feedback on the servos, so platform-level tracking error is uncharacterized. The PX4 IMU is the only physical measurement in the loop. This is an honest limitation: we measure the output (attitude) but not the intermediate stage (platform pose). Adding encoders is a possible future improvement.

> 역기구학을 명령하나 서보에 인코더 피드백이 없어 플랫폼 수준의 추종 오차는 미규명. PX4 IMU가 루프 내 유일한 물리 측정. 출력(자세)은 측정하나 중간 단계(플랫폼 pose)는 미측정. 인코더 추가는 향후 개선 가능.

**"What is the mounting bias physically? Can you eliminate it?"** → Slide 8.
The ~1.8° bias comes from the physical alignment of the FC on the platform top plate. It is consistent within a run (std ~0.2°) and can be calibrated out in post-processing, which we do. The physical mount could also be shimmed or re-aligned, but for HILS validation purposes, software correction is sufficient as long as the bias is stable.

> ~1.8° 바이어스는 플랫폼 상판 위 FC의 물리적 정렬에서 기인. 실험 내 일관적(std ~0.2°)이며, 후처리에서 보정 가능(실제 수행). 물리적 마운트 조정도 가능하나, HILS 검증 목적에서는 바이어스가 안정적이면 소프트웨어 보정으로 충분.

**"How repeatable are the results across runs?"** → Slides 8 + 11.
We have 5 baseline runs (#5–#9). Within each run, bias std is ~0.2°. Across runs, the bias estimate varies by up to 0.8°, likely due to thermal drift and minor re-seating between runs. Full repeatability characterization is listed as a next step.

> 5회 기본 실험(#5–#9) 보유. 실험 내 바이어스 std ~0.2°. 실험 간 바이어스 추정치는 최대 0.8° 차이, 열적 드리프트 및 미세 재장착에 기인. 완전한 반복성 분석은 향후 과제.

**"Can this extend to quadrotors or other vehicles?"** → Slide 11.
The architecture is vehicle-agnostic. Only the X-Plane aircraft model and the PID tuning change. The Stewart platform is a general 6-DOF motion device. PX4 itself supports both fixed-wing and multirotor, so the same hardware setup can validate different vehicle classes.

> 아키텍처는 기체 비종속적. X-Plane 기체 모델과 PID 튜닝만 변경. 스튜어트 플랫폼은 범용 6-DOF 모션 장치. PX4 자체가 고정익/멀티로터 지원하므로 동일 하드웨어로 다양한 기체 검증 가능.
