# Conference Presentation Script

English primary, Korean translation indented below each paragraph.
Slide numbers match the current deck (12 slides).

---

## Slide 1 — Title / One-line contribution

Hello, I'm presenting our work on a physically closed-loop HILS system using a Stewart platform for vehicle attitude-control validation.

> 안녕하세요, 스튜어트 플랫폼을 활용한 물리적 폐루프 HILS 자세제어 검증에 대해 발표하겠습니다.

In one sentence, our contribution is this: we connect X-Plane, a real Stewart platform, and a real PX4 flight controller so that simulated dynamics drive physical motion, and measured physical attitude feeds back into the simulator loop.

> 한마디로, X-Plane 시뮬레이터, 실물 스튜어트 플랫폼, 실물 PX4 비행 제어기를 하나로 연결했습니다. 시뮬레이션이 플랫폼을 물리적으로 움직이고, 그 움직임을 PX4가 감지해서 다시 시뮬레이터로 피드백하는 구조입니다.

Today I'll cover motivation, architecture, experiment design, quantitative results, limitations, and next steps.

> 오늘 발표 순서는 연구 동기, 시스템 구조, 실험 설계, 정량적 결과, 한계점, 그리고 향후 계획 순입니다.

`>> NEXT SLIDE`

## Slide 2 — Why add physical motion to HILS?

Conventional HILS is powerful, but in many setups the flight controller receives synthesized sensor signals, and attitude is checked mainly through screens and logs.

> 기존 HILS도 강력한 검증 수단이지만, 대부분의 경우 비행 제어기에 가상 센서 신호를 주입하고, 자세 확인은 화면이나 로그에 의존하게 됩니다.

Our motivation is to add physical realism: the FC's IMU should experience real inertial excitation, not only virtual values.

> 저희가 주목한 점은 물리적 현실감입니다. 비행 제어기의 IMU가 가상 값이 아니라 실제 관성 운동을 겪도록 하자는 것이 출발점입니다.

This matters beyond fixed-wing aircraft. Any vehicle class where attitude dynamics are critical — quadcopters, satellites, missiles — can benefit from the same physically closed-loop validation concept.

> 그리고 이 접근법은 고정익 항공기에만 해당되는 이야기가 아닙니다. 쿼드콥터, 인공위성, 유도탄 등 자세 동역학이 중요한 모든 비행체에 동일한 개념을 적용할 수 있습니다.

So this work asks: Can we build a stable, repeatable, physically closed-loop testbed, and quantify tracking error and latency?

> 그래서 본 연구의 핵심 질문은 이것입니다. 안정적이고 반복 가능한 물리 폐루프 시험 환경을 만들 수 있는가, 그리고 추종 오차와 지연을 정량적으로 평가할 수 있는가.

`>> NEXT SLIDE`

## Slide 3 — What is new here?

The key differentiation is not just hardware presence, but physical sensing in the loop.

> 핵심적인 차이는 하드웨어가 있다는 것 자체가 아니라, 루프 안에서 실제 물리량을 센싱한다는 점입니다.

In our system, simulated attitude is realized by platform motion, and the PX4 estimator responds to that real motion. This creates a more realistic closed-loop behavior within platform limits.

> 저희 시스템에서는 시뮬레이션 자세를 플랫폼이 물리적으로 재현하고, PX4 추정기가 그 실제 움직임에 반응합니다. 이를 통해 플랫폼 한계 범위 내에서 보다 현실적인 폐루프 거동이 구현됩니다.

At the same time, I want to be explicit: this does not replace conventional HILS. Workspace limits and missing aerodynamic force reproduction mean it is a complementary validation tool.

> 다만 분명히 말씀드리면, 이 시스템이 기존 HILS를 대체하는 것은 아닙니다. 작업 영역의 한계와 공기역학적 하중 미재현을 고려하면, 기존 방법을 보완하는 검증 도구입니다.

Importantly, the architecture is vehicle-agnostic. The Stewart platform provides general 6-DOF motion; only the simulator model and controller tuning need to change to validate a different vehicle class — from multirotor UAVs to spacecraft attitude simulators.

> 또한 이 구조는 특정 기체에 종속되지 않습니다. 스튜어트 플랫폼은 범용 6자유도 모션 장치이므로, 시뮬레이터 모델과 제어기 튜닝만 바꾸면 멀티로터 UAV부터 위성 자세 시뮬레이터까지 다양한 기체를 검증할 수 있습니다.

`>> NEXT SLIDE`

## Slide 4 — System in action

[Show hardware photo and/or demo video]

Here you can see the actual system operating. The Stewart platform physically moves to match the simulated aircraft attitude, and the PX4 mounted on top senses that motion in real time.

> 실제 시스템이 작동하는 모습입니다. 스튜어트 플랫폼이 시뮬레이션 항공기의 자세에 맞춰 움직이고, 상판에 장착된 PX4가 이 움직임을 실시간으로 감지합니다.

`>> NEXT SLIDE`

## Slide 5 — How the physical closed loop works

Here is the full dataflow.

> 전체 데이터 흐름을 보겠습니다.

[Show Fig. 1: System architecture/data flow with numbered callouts]

Step-by-step:

1. X-Plane publishes vehicle state to the Python host over UDP.
2. Host converts state to pose commands for the Stewart platform.
3. PX4 physically senses platform motion through its onboard IMU.
4. PX4 attitude returns to the host; host injects control back into X-Plane.

> 단계별로 설명드리면:
>
> 1. X-Plane이 UDP로 비행체 상태를 Python 호스트에 전달합니다.
> 2. 호스트가 이를 스튜어트 플랫폼의 자세 명령으로 변환합니다.
> 3. PX4가 자체 IMU로 플랫폼의 물리적 움직임을 감지합니다.
> 4. PX4의 자세 데이터가 호스트를 거쳐 X-Plane에 제어 입력으로 주입됩니다.

So the loop is fully closed across software, mechanics, sensing, and control I/O.

> 이렇게 소프트웨어, 기구부, 센싱, 제어 I/O가 모두 연결된 완전한 폐루프가 구성됩니다.

`>> NEXT SLIDE`

## Slide 6 — What we tested and how we measured it

The platform's neutral position is set at z=+20 mm to maximize the available rotation range.

> 플랫폼의 중립 위치는 회전 가용 범위를 최대화하기 위해 z=+20 mm으로 설정했습니다.

Our baseline scenario is a trim transition: pitch setpoint goes from 0 degrees to +5 degrees, holds, then returns.

> 기본 시나리오는 트림 전환으로, 피치 설정값을 0도에서 +5도로 올렸다가 유지한 뒤 복귀시킵니다.

Each run starts from the same simulator initial condition and is repeated multiple times for statistics.

> 매 실험은 동일한 시뮬레이터 초기 조건에서 출발하고, 통계적 분석을 위해 여러 차례 반복했습니다.

Outputs per run include X-Plane attitude, PX4 attitude, command/ACK timing, and post-processed metrics. This baseline was selected because it is simple, repeatable, and directly supports quantitative evaluation.

> 각 실험에서 X-Plane 자세, PX4 자세, 명령/ACK 타이밍, 후처리 메트릭을 기록합니다. 이 시나리오를 선택한 이유는 단순하고 반복이 용이하며, 정량적 평가에 직접 활용할 수 있기 때문입니다.

`>> NEXT SLIDE`

## Slide 7 — Result 1: Closed-loop tracking is achieved

[Show Fig. 2: pitch response overlay]

In the overlay, PX4 pitch follows the commanded transition with expected lag. Raw error includes a near-constant offset component.

> 그래프를 보시면, PX4 피치가 예상되는 시간 지연을 두고 명령을 추종하는 것을 확인할 수 있습니다. 원시 오차에는 거의 일정한 오프셋 성분이 포함되어 있습니다.

That offset is not random instability. Evidence suggests a mounting/frame bias component, so we also evaluate bias-corrected error.

> 이 오프셋은 무작위적인 불안정성이 아닙니다. 센서 장착 정렬에서 오는 바이어스로 판단되며, 이를 보정한 오차도 함께 평가했습니다.

Main point: closed-loop tracking is achieved and quantifiable, with error structure that is explainable.

> 핵심은, 폐루프 추종이 실현되었고 정량적으로 평가할 수 있으며, 오차의 원인도 설명 가능하다는 것입니다.

`>> NEXT SLIDE`

## Slide 8 — Result 2: Residual error is interpretable

[Show Fig. 3: pitch error + Fig. 4: mounting bias evidence]

A substantial portion of the observed pitch error behaves like a mounting bias. Correcting for it changes the interpretation of fidelity significantly.

> 관측된 피치 오차의 상당 부분이 장착 바이어스 형태로 나타납니다. 이를 보정하면 충실도에 대한 해석이 크게 달라집니다.

This is important scientifically: the dominant residual error is not random instability. A fixed bias component must be separated from dynamic tracking fidelity.

> 이 점이 중요한 이유는, 주된 잔여 오차가 무작위적 불안정이 아니라는 것입니다. 고정 바이어스 성분과 동적 추종 충실도를 분리해서 봐야 정확한 평가가 가능합니다.

#! wrong content. slide is meant to explain the source of error as a mounting bias. and proving it by comparing error tendency btw loose mounting run and firm mounting run. which loose mounting runs show error shows tedency to increase small bit. when the firm mounting run shows relatively flat and monotonic error accross the board in contrast. this shows pitch error have relationship with mounting quality of RC plane used and the stewart platform. fyi, rc plane was mounted using masking tape in loose mounting run, and was glued down with hot glue and bounded to upper platform with paracord tightly in firm mounting run

`>> NEXT SLIDE`

## Slide 9 — Result 3: Latency sets the fidelity limit

[Show Fig. 5(a): latency time series + Fig. 5(b): latency distribution]

We break timing into sample age, serial RTT, and end-to-end delays. Key numbers: serial RTT about 22 ms, PX4 rx-to-ACK about 41 ms, X-Plane rx-to-ACK about 51 ms.

> 타이밍을 sample age, serial RTT, 종단간 지연으로 분해해서 분석했습니다. 주요 수치로는 serial RTT 약 22 ms, PX4 수신-ACK 간 약 41 ms, X-Plane 수신-ACK 간 약 51 ms입니다.

Latency is not a side note; it directly explains observed phase lag and bounds achievable closed-loop fidelity.

> 지연은 단순한 부수 정보가 아닙니다. 관측된 위상 지연의 직접적인 원인이며, 달성 가능한 폐루프 충실도의 상한을 결정합니다.

In baseline runs, saturation was minimal or absent, so these results primarily represent nominal-region behavior.

> 기본 실험에서는 포화가 거의 발생하지 않았기 때문에, 이 결과는 정상 작동 영역에서의 거동을 대표합니다.

#! need to explain what serial RTT, sample age, end to end, etc means before discussion. lacks information about sim system performace. such as bandwith phase lag, Jitter, saturation, bias/drift

`>> NEXT SLIDE`

## Slide 10 — What this platform does not claim

I want to clearly state limitations:

> 이 시스템의 한계를 분명히 짚겠습니다:

- Finite workspace: large multi-axis commands reduce the feasible operating region.
- No physical aerodynamic load: motion cues are reproduced, not full external-force physics.
- Platform-level actuation limits: timing, hysteresis, and actuator constraints shape response fidelity.
- Not a flight replacement: this complements conventional HILS and real flight testing.

> - 유한한 작업 영역: 대각도 다축 명령에서는 구동 가능 영역이 줄어듭니다.
> - 공기역학적 하중 미재현: 모션 큐는 재현하지만 외력은 구현하지 않습니다.
> - 플랫폼 구동 한계: 타이밍, 히스테리시스, 액추에이터 제약이 응답 충실도에 영향을 줍니다.
> - 비행 시험 대체 불가: 기존 HILS와 실제 비행 시험을 보완하는 역할입니다.

The correct claim is: a practical, measurable, physically informed pre-flight validation layer.

> 저희가 주장하는 바는 명확합니다. 실용적이고, 측정 가능하며, 물리적 근거를 갖춘 비행 전 검증 수단이라는 것입니다.

#! largely uneseccery and harmful even. doesn't warrant separate slide. should have been breifly mentioned in front or last shortly

`>> NEXT SLIDE`

## Slide 11 — Takeaways

Four takeaways:

> 핵심 내용을 정리하겠습니다.

First, integration: physically closed-loop HILS is successfully implemented with X-Plane, Stewart platform, and PX4.

> 첫째, 통합입니다. X-Plane, 스튜어트 플랫폼, PX4를 연동한 물리적 폐루프 HILS를 성공적으로 구현했습니다.

Second, fidelity: closed-loop tracking is achieved and can be evaluated quantitatively.

> 둘째, 충실도입니다. 폐루프 자세 추종을 달성했고, 정량적으로 평가할 수 있음을 보였습니다.

Third, transparency: bias and latency are measured and used to define the platform's true fidelity, not hidden.

> 셋째, 투명성입니다. 바이어스와 지연을 숨기지 않고 측정하여, 플랫폼의 실제 충실도를 명확히 제시했습니다.

Fourth, generality: the architecture is vehicle-agnostic. The same testbed extends to quadcopters, satellites, missiles, and any vehicle where attitude dynamics validation matters.

> 넷째, 범용성입니다. 이 구조는 특정 기체에 종속되지 않습니다. 쿼드콥터, 인공위성, 유도탄 등 자세 동역학 검증이 필요한 모든 비행체로 확장할 수 있습니다.

Near-term next steps: time-aligned error analysis, repeatability/hysteresis characterization, stress tests near saturation boundaries. Longer term, we plan to demonstrate cross-vehicle generality by validating different vehicle types on the same hardware.

> 단기 과제로는 시간 정렬 기반 오차 분석, 반복성 및 히스테리시스 특성화, 포화 경계 부근 스트레스 시험을 계획하고 있습니다. 장기적으로는 동일 하드웨어에서 다양한 기체를 검증하여 범용성을 실증할 계획입니다.

Thank you. I welcome questions.

> 이상으로 발표를 마치겠습니다. 질문 있으시면 말씀해 주십시오.

#! should add ending slide that only shows the slides id over.

`>> BACKUP SLIDE (if needed)`

## Slide 12 — Q&A backup

Use this slide only if a specific question comes up. Otherwise stay on Slide 11.

> 특정 질문이 나올 때만 이 슬라이드를 사용합니다. 그 외에는 슬라이드 11에 머물러 주세요.

---

## Optional live delivery cues

- If time is short, compress Slides 5 and 6 into one minute total.
- Spend most technical depth on Slides 5, 7, and 9.
- Slides 4 (demo footage) is for audience engagement — keep it brief, 15–20 seconds.

---

## Q&A prepared answers

**"Why z=+20 mm as the neutral position?"** → Slide 12, left panel with proof figure.
The rotation range is a function of z height. At z=+20 mm we get ~20° of rotation; at z=0 mm only ~10°. This maximizes the usable attitude envelope.

> 회전 가용 범위가 z 높이에 따라 달라집니다. z=+20 mm에서는 약 20°, z=0 mm에서는 약 10°밖에 안 됩니다. 자세 구현 범위를 최대로 확보하기 위한 선택입니다.

**"Why not just conventional HILS?"** → Slides 3 + 10.
We add real inertial excitation that sensor synthesis cannot provide, while honestly acknowledging workspace and aerodynamic reproduction limits.

> 센서 신호 합성만으로는 줄 수 없는 실제 관성 운동을 제공하면서도, 작업 영역과 공기역학적 재현의 한계는 솔직하게 인정하고 있습니다.

**"How do you handle saturation?"** → Slides 6 + 10 + 11.
Baseline runs operate at ±5° out of ~20° available. Saturation was minimal. Stress tests near saturation boundaries are a planned next step.

> 기본 실험은 약 20° 가용 범위 중 ±5°에서 운용했고, 포화는 거의 없었습니다. 포화 한계 부근 스트레스 시험은 향후 과제입니다.
#! no saturation observed in any run, so omitted. should have mentioned it before.

**"What about yaw?"** → Slide 6.
Yaw is in the loop, but the trim-transition scenario primarily excites pitch. Yaw tracking and bias alignment are handled in the code but not the focus of this presentation.

> Yaw도 루프에 포함되어 있지만, 트림 전환 시나리오에서는 주로 피치가 여기됩니다. Yaw 추종과 바이어스 정렬은 코드에 구현되어 있으나, 이번 발표에서는 다루지 않습니다.

**"What's the effective bandwidth of this system?"** → Slides 5 + 9.
The host ticks at ~50 Hz. End-to-end latency from X-Plane receive to Stewart ACK is ~51 ms. This means the system is suited for low-frequency maneuvers like trim transitions; high-frequency dynamic content is filtered by the latency chain. Increasing bandwidth would require reducing serial overhead or moving to a faster transport.

> 호스트 루프가 약 50 Hz로 동작하고, X-Plane 수신부터 Stewart ACK까지 종단간 지연이 약 51 ms입니다. 따라서 트림 전환 같은 저주파 기동에 적합하며, 고주파 동역학 성분은 지연에 의해 걸러집니다. 대역폭을 높이려면 시리얼 오버헤드를 줄이거나 더 빠른 통신 수단으로 전환해야 합니다.

**"The platform is open-loop — how do you trust pose accuracy?"** → Slide 10.
We command inverse kinematics but have no encoder feedback on the servos, so platform-level tracking error is uncharacterized. The PX4 IMU is the only physical measurement in the loop. This is an honest limitation: we measure the output (attitude) but not the intermediate stage (platform pose). Adding encoders is a possible future improvement.

> 역기구학으로 명령을 내리지만, 서보에 인코더 피드백이 없어서 플랫폼 자체의 추종 오차는 알 수 없습니다. PX4 IMU가 루프 안에서 유일한 물리 측정 수단입니다. 즉, 최종 출력인 자세는 측정하지만 중간 단계인 플랫폼 포즈는 측정하지 못합니다. 인코더 추가는 향후 개선 방안으로 고려하고 있습니다.
!# should mention original goal was to make stewart platfrom as cheap as possible, which lead me to choose cheap rc aircraft grade servo. which explains lack of feedback. but error was measured with digital inclinometer to about 2.1 deg and 1.5 deg (roll, pitch)

**"What is the mounting bias physically? Can you eliminate it?"** → Slide 8.
The ~1.8° bias comes from the physical alignment of the FC on the platform top plate. It is consistent within a run (std ~0.2°) and can be calibrated out in post-processing, which we do. The physical mount could also be shimmed or re-aligned, but for HILS validation purposes, software correction is sufficient as long as the bias is stable.

> 약 1.8°의 바이어스는 플랫폼 상판 위에 FC를 장착할 때의 물리적 정렬 오차에서 비롯됩니다. 실험 내에서는 일관되게 나타나고(표준편차 약 0.2°), 후처리로 보정할 수 있으며 실제로 보정하고 있습니다. 마운트를 물리적으로 조정할 수도 있지만, HILS 검증 목적으로는 바이어스가 안정적인 한 소프트웨어 보정으로 충분합니다.

**"How repeatable are the results across runs?"** → Slides 8 + 11.
We have 5 baseline runs (#5–#9). Within each run, bias std is ~0.2°. Across runs, the bias estimate varies by up to 0.8°, likely due to thermal drift and minor re-seating between runs. Full repeatability characterization is listed as a next step.

> 기본 실험을 5회(#5~#9) 수행했습니다. 실험 내 바이어스의 표준편차는 약 0.2°이고, 실험 간에는 최대 0.8°까지 차이가 나는데, 이는 열적 드리프트와 미세한 재장착 차이 때문으로 보입니다. 체계적인 반복성 분석은 향후 과제입니다.

**"Can this extend to quadrotors or other vehicles?"** → Slide 11.
The architecture is vehicle-agnostic. Only the X-Plane aircraft model and the PID tuning change. The Stewart platform is a general 6-DOF motion device. PX4 itself supports both fixed-wing and multirotor, so the same hardware setup can validate different vehicle classes.

> 시스템 구조가 특정 기체에 종속되지 않습니다. X-Plane의 기체 모델과 PID 튜닝만 바꾸면 됩니다. 스튜어트 플랫폼은 범용 6자유도 모션 장치이고, PX4 자체가 고정익과 멀티로터를 모두 지원하므로, 동일한 하드웨어 구성으로 다양한 기체를 검증할 수 있습니다.
