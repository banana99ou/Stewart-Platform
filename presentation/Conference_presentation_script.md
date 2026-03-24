# Conference Presentation Script

Korean spoken script first (for reading aloud), English reference below (for LLM context).
Slide numbers match the current deck (17 slides).

---

# 발표 대본 (Korean Spoken Script)

## 슬라이드 1 — 제목

안녕하세요, 국민대학교 미래모빌리티제어연구실 석사과정 정현용입니다. 오늘은 스튜어트 플랫폼 기반 물리적 폐루프 HILS를 이용해, PX4 추정기를 실제 관성 운동으로 어떻게 검증할 수 있는지 말씀드리겠습니다.

간단히 말씀드리면, X-Plane 시뮬레이터, 실물 스튜어트 플랫폼, 실물 PX4 비행 제어기를 하나의 루프로 연결했습니다. 시뮬레이터가 플랫폼을 물리적으로 움직이고, PX4가 그 움직임을 감지해서, 그 측정값이 다시 시뮬레이션으로 돌아가는 구조입니다.

핵심 질문은, PX4의 탑재 추정기가 합성 IMU 데이터가 아닌 실제 물리적 운동에 의해 구동될 때 어떻게 반응하는가입니다. 이 시스템은 바로 그것을 검증하기 위한 플랫폼입니다.

연구 동기부터 시스템 구성, 실험 내용, 측정 결과, 향후 계획까지 순서대로 말씀드리겠습니다.

`>> NEXT SLIDE`

## 슬라이드 2 — 왜 물리적 운동을 HILS에 추가하는가?

그러면 왜 HILS에 물리적 운동을 추가해야 할까요?

기존 HILS도 강력합니다. 하지만 대부분의 구성에서 비행 제어기는 합성된 센서 데이터를 받고, 자세는 화면이나 로그 파일로 확인하죠. IMU가 실제로 움직이지는 않습니다.

저희가 원한 것은 물리적 현실감입니다. 비행 제어기의 IMU가 실제 관성 운동을 겪도록 하자는 것이 출발점이었습니다.

이 접근법 자체는 고정익에만 한정되지 않습니다 — 자세 동역학이 중요한 비행체라면 같은 원리를 적용할 수 있습니다. 다만 이번 연구에서는 고정익을 첫 번째 실증 대상으로 선택했습니다.

오늘 발표에서는, 이 물리 폐루프 시험 환경이 실제로 안정적으로 동작하고, 추종 성능을 정량화할 수 있음을 실험 데이터로 보여드리겠습니다.

`>> NEXT SLIDE`

## 슬라이드 3 — 무엇이 새로운가?

그러면 구체적으로 뭐가 새로운 걸까요?

핵심은 PX4의 추정기가 합성 신호가 아니라 실제 물리적 움직임에 반응한다는 점입니다. 기존 HILS에서는 IMU 데이터가 소프트웨어로 생성되지만, 이 시스템에서는 실제 관성 운동을 추정기가 직접 경험합니다. 이것이 검증의 질을 바꿉니다.

다만 미리 말씀드리면, 이 시스템이 기존 HILS나 실제 비행 시험을 대체하지는 않습니다. 플랫폼의 작업 영역은 유한하고, 구동 한계가 있고, 공기역학적 하중도 재현하지 못합니다. 보완적 검증 수단으로 보시면 됩니다 — 다만 순수 소프트웨어 루프로는 얻을 수 없는 물리적 근거를 추가한다는 점이 핵심입니다.

스튜어트 플랫폼은 이 시스템에서 시뮬레이터와 비행 제어기 사이의 물리적 연결 고리 역할을 합니다 — 소프트웨어 자세 명령을 실제 6자유도 운동으로 바꿔서 PX4 IMU가 실제로 움직이게 합니다. 구체적인 아키텍처는 바로 다음에 보여드리겠습니다.

`>> NEXT SLIDE`

## 슬라이드 4 — 시스템 구동 모습

[사진 및/또는 데모 영상]

실제 시스템이 작동하는 모습입니다. 스튜어트 플랫폼이 시뮬레이션 자세에 맞춰 움직이고, 위에 탑재된 PX4가 그 움직임을 실시간으로 감지하고 있습니다.

`>> NEXT SLIDE`

## 슬라이드 5 — 물리적 폐루프 동작 원리

이제 폐루프가 실제로 어떻게 동작하는지 보겠습니다.

[그림 1: 시스템 구조/데이터 흐름도 — 번호 1~4 표시]

크게 네 단계입니다.

1. X-Plane이 UDP로 비행체 상태를 Python 호스트에 보냅니다.
2. 호스트가 이를 스튜어트 플랫폼의 자세 명령으로 변환합니다.
3. 플랫폼 위의 PX4가 자체 IMU로 물리적 움직임을 감지합니다.
4. PX4 자세 추정값이 호스트로 돌아오면, 호스트의 PID 제어기가 조종면 명령을 산출하여 X-Plane에 주입합니다.

소프트웨어, 기구부, 물리 센싱, 제어 I/O가 하나의 완전한 루프로 연결되는 구조입니다. 이 블록들을 하나씩 살펴보겠습니다.

`>> NEXT SLIDE`

## 슬라이드 6 — 블록 1: X-Plane 시뮬레이터

[구조도에서 1번 블록 강조]

루프의 출발점은 X-Plane입니다. 여기서 동역학이 연산되고, 트림 전환 같은 시나리오를 반복 실행할 수 있는 환경을 제공합니다. 기체 상태는 UDP로 호스트에 전송됩니다.

기존에는 X-Plane 안에서 루프가 닫혔지만, 이 구조에서는 PX4가 측정한 자세를 호스트의 PID 제어기가 받아 조종면 명령으로 변환하고, 그 명령이 X-Plane에 주입됩니다. 그래서 시뮬레이션 결과 자체가 물리적 루프의 영향을 받게 됩니다.

`>> NEXT SLIDE`

## 슬라이드 7 — 블록 2: Python 호스트

[구조도에서 2번 블록 강조]

Python 호스트는 시스템 전체의 통합 허브입니다.

X-Plane에서 UDP로 기체 상태를, PX4에서 MAVLink/USB로 자세 추정값을 받습니다. 시뮬레이터 상태를 플랫폼 pose 명령으로 변환하고, 모든 스트림에 공통 타임스탬프를 부여해 로깅합니다.

변환된 명령은 시리얼로 플랫폼에 전송합니다. PX4 자세 추정값은 호스트 내장 PID 제어기의 피드백으로 사용되며, PID가 산출한 조종면 명령이 UDP로 X-Plane에 주입되어 루프를 완성합니다. 타이밍이나 인터페이스, 로깅을 이해하려면 이 블록이 핵심입니다.

`>> NEXT SLIDE`

## 슬라이드 8 — 블록 3: Stewart 플랫폼 구동

[구조도에서 3번 블록 강조]

세 번째는 스튜어트 플랫폼입니다. 소프트웨어 명령을 실제 물리적 모션으로 바꾸는 블록입니다.

호스트에서 받은 pose 명령을 플랫폼의 구동 범위 안에서 포화한 뒤 실제로 플랫폼을 움직입니다. 처리가 끝나면 ACK, 처리 시간, 범위 포화 여부와 스케일링 계수 α를 호스트에 반환합니다. 호스트-플랫폼 간 시리얼 명령-ACK 왕복 시간이 약 22 ms이고 서보 PWM 주기가 50 Hz(20 ms)이므로, 전체 루프는 약 50 Hz 근처에서 동작합니다.

이 블록을 지나는 순간부터 PX4가 경험하는 것은 합성 신호가 아니라 실제 관성 운동입니다.

`>> NEXT SLIDE`

## 슬라이드 9 — 블록 4: PX4 센싱 및 귀환

[구조도에서 4번 블록 강조]

마지막으로 PX4가 물리 루프를 완성합니다.

플랫폼 위에 장착된 PX4의 온보드 IMU가 실제 운동을 감지하고, 자체 추정기로 자세를 실시간 산출합니다. 이 추정값이 MAVLink/USB를 통해 호스트로 전달되면, 호스트의 PID 제어기가 조종면 명령을 산출해 X-Plane에 주입하고 루프를 완성합니다.

검증 대상은 소프트웨어 모사 추정기가 아니라, 실제 하드웨어에서 실제 운동에 반응하는 PX4 추정기 그 자체입니다.

`>> NEXT SLIDE`

## 슬라이드 10 — 실험 설계 및 측정 방법

자, 그러면 실제로 무엇을 실험했는지 보겠습니다.

플랫폼 중립 위치는 z = +20 mm으로 설정했습니다. 회전 가용 범위가 최대가 되는 높이입니다.

기본 시나리오는 트림 전환입니다. 피치를 0도에서 +5도로 올렸다가, 유지한 뒤, 다시 돌아옵니다. 모든 실험은 동일한 X-Plane 초기 조건인 호놀룰루 PHNL 10해리 어프로치에서 시작합니다.

각 실험 전에 스크립트로 설정한 유예 기간 동안 시스템을 안정화하고, 실험 내내 스튜어트 플랫폼의 포화 플래그를 감시하여 구동 가능 영역 안에 있는지 확인합니다. 조건별로 9회 반복하여 통계적 유의미성을 확보했습니다.

각 실험에서 X-Plane 자세, PX4 자세, 플랫폼 자세 명령, 명령-ACK 타이밍, 후처리 메트릭을 기록합니다. 단순하고 반복 가능하며, 정량 분석에 바로 쓸 수 있기 때문에 이 시나리오를 선택했습니다.

시스템 자체의 지연, 바이어스, 장착 효과를 먼저 분리해 파악해야 하므로, 제어된 조건의 깨끗한 기본 시나리오에서 시작합니다. 이 기준선이 있어야 이후 교란이나 포화 실험에서 시스템 오차와 시나리오 오차를 구분할 수 있습니다.

`>> NEXT SLIDE`

## 슬라이드 11 — 결과 1: 폐루프 추종 달성

이제 결과를 보겠습니다. 먼저 추종 성능입니다.

[그림 2: 피치 응답 오버레이]

보시면 PX4 피치가 명령 전환을 잘 따라가고 있습니다. 트림 전환은 저주파 기동이라 위상 지연이 그래프에서 두드러지지 않으며, 지연에 대한 정량 분석은 슬라이드 13에서 다루겠습니다. 주목할 점은 보정 전 오차에 거의 일정한 오프셋이 포함되어 있다는 것입니다.

이 오프셋은 무작위적 불안정이 아닙니다. 약 +1.8도로 실험 간에 일관되게 나타나며, 장착 바이어스로 판단됩니다. 그래서 보정 전 오차와 바이어스 보정 후 오차를 둘 다 평가했습니다.

결론적으로, 폐루프 추종이 작동하고, 수치로 평가할 수 있으며, 오차 구조가 물리적으로 설명이 됩니다.

`>> NEXT SLIDE`

## 슬라이드 12 — 결과 2: 장착 품질이 주된 오차의 원인

바이어스가 있다면, 어디서 오는 걸까요? 장착 품질이 원인이라는 가설을 두 가지 근거로 확인했습니다.

[그림 3: 피치 오차 + 그림 4: 장착 바이어스 근거]

첫 번째 근거. 장착 조건을 바꾸면 바이어스의 거동이 달라집니다. 느슨한 장착(접착 테이프만 사용)에서는 관성 하중에 의해 FC가 서서히 밀리면서 바이어스가 시간에 따라 증가합니다. 단단한 장착(구조용 접착제 + 기계적 결속)으로 바꾸면 이 드리프트가 사라집니다.

두 번째 근거. 단단한 장착 조건에서 남는 바이어스는 실험 내내 거의 일정합니다. 이는 남은 오차가 동적 불안정이 아니라, 장착 시 생긴 고정된 정렬 오차라는 뜻입니다.

두 근거를 종합하면, 주된 피치 오차의 원인은 제어 시스템이 아니라 장착 품질입니다. 장착을 잡으면 오차가 보정 가능한 정적 바이어스로 수렴하며, 후처리 스크립트가 이 바이어스를 자동으로 추정하고 보정합니다.

`>> NEXT SLIDE`

## 슬라이드 13 — 결과 3: 지연 특성화 및 시스템 성능

추종 정확도는 봤는데, 그러면 이 시스템이 얼마나 빠르게 반응할 수 있을까요? 실질적으로 충실도를 제한하는 건 바로 이 부분입니다.

[그림 5(a): 지연 시계열 + 그림 5(b): 지연 분포]

타이밍 지표는 네 가지로 나뉩니다. 첫째, X-Plane 샘플 에이지는 호스트가 시뮬레이터 데이터를 사용하는 시점에서 그 데이터가 얼마나 오래된 것인지로 평균 약 29 ms입니다. 둘째, PX4 샘플 에이지는 PX4 자세 추정값이 호스트에 도달했을 때의 경과 시간으로 평균 약 19 ms입니다. 셋째, 시리얼 RTT는 호스트가 스튜어트 플랫폼에 명령을 보내고 ACK를 받기까지의 왕복 시간으로 평균 약 22 ms입니다. 넷째, 종단간 지연은 X-Plane 데이터가 호스트에 도착한 시점부터 플랫폼이 동작을 완료하고 ACK가 돌아오기까지의 전체 시간으로 평균 약 51 ms입니다.

이게 중요한 이유는, 이 지연 사슬이 동적 추종 한계를 정하고 유효 bandwidth를 PWM 주기나 호스트 tick만으로 추정할 수 없다는 점을 보여주기 때문입니다.

정리하면, 호스트-플랫폼 갱신 루프는 약 50 Hz로 동작하고 대표 baseline run에서 종단간 지연은 평균 약 51 ms였습니다. 다만 타이밍 분포 폭이 충분히 작다고 보기는 어려우므로, 현재 데이터만으로 유효 bandwidth를 직접 주장하지는 않고 전용 sine-sweep 측정으로 정량화할 예정입니다. 또한 z=+20 mm 중립 위치에서의 가용 회전 범위 안에서 baseline 명령은 대부분 정상 영역에 머물렀고, baseline run들의 포화 비율도 0-1% 수준이었습니다. 그리고 앞 슬라이드에서 봤듯이 잔류 바이어스는 대체로 안정적이며 후처리로 보정 가능합니다.

`>> NEXT SLIDE`

## 슬라이드 14 — 핵심 정리

정리하겠습니다. 세 가지 핵심 사항입니다.

첫째, 통합. X-Plane, 스튜어트 플랫폼, PX4를 하나의 물리적 폐루프 HILS로 구현했습니다.

둘째, 충실도. 바이어스 보정 후 피치 추종 오차 RMSE가 약 0.46도이고 종단간 지연은 약 51 ms로, 폐루프 추종 성능을 정량적으로 제시할 수 있습니다.

셋째, 시스템 특성화를 통해 향후 확장의 기반을 확보했습니다. 지연, 바이어스, 작업 영역의 정량적 한계를 확인함으로써, 교란 실험이나 다른 기체 유형으로의 확장을 위한 기준선을 마련했습니다.

향후 계획으로는, 단기적으로 시간 정렬 오차 분석, sine-sweep 기반 bandwidth 측정, 반복성 특성화, 포화 경계 부근 스트레스 시험을 예정하고 있습니다. 장기적으로는 멀티로터 등 다른 기체로 확장하여 아키텍처의 범용성을 실증하는 것이 목표입니다.

`>> NEXT SLIDE`

## 슬라이드 15 — 마무리

감사합니다. 질문 있으시면 말씀해 주세요.

`>> BACKUP SLIDE (if needed)`

## 슬라이드 16 — Q&A 백업

특정 질문이 나올 때만 이 슬라이드를 사용합니다. 그 외에는 슬라이드 15에 머물러 주세요.

---

## 발표 운용 참고사항

- 시간 부족 시 슬라이드 6~9(블록 상세)를 빠르게 넘기고, 슬라이드 5 개요에서 핵심만 전달.
- 기술적 심화는 슬라이드 7(호스트), 11(추종), 13(지연)에 집중.
- 슬라이드 4 (데모 영상)는 청중 몰입용 — 15~20초 이내.

---
---

# English Reference Script

The English version below is the reference/source text for the Korean spoken script above.
It also contains Q&A prepared answers for anticipated questions.

---

## Slide 1 — Title / One-line contribution

Hello, I'm Hyeon-yong Jeong. Today I'll be talking about our physically closed-loop HILS testbed built around a Stewart platform, with a focus on validating the PX4 estimator under real inertial motion.

In short — we connected X-Plane, a real Stewart platform, and a real PX4 flight controller into one loop. The simulator drives physical motion, the PX4 senses it, and that measurement feeds right back into the simulation.

The core question is whether the PX4's onboard estimator responds differently when driven by real physical motion versus synthesized IMU data — and this platform is built to answer that.

I'll walk you through the motivation, how we built it, what we tested, what we measured, and where we go next.

## Slide 2 — Why add physical motion to HILS?

So why add physical motion to HILS?

Conventional HILS is powerful. But in most setups, the flight controller gets synthesized sensor data. You check attitude on a screen or in a log file. The IMU never actually moves.

What we wanted was physical realism — to make the FC's IMU experience real inertial motion, not just numbers fed through a wire.

This approach is not limited to fixed-wing aircraft — any vehicle where attitude dynamics matter could use the same principle. We chose fixed-wing as the first demonstration case.

In this talk, we show with experimental data that this physically closed-loop testbed works stably and that tracking performance is quantifiable.

## Slide 3 — What is new here?

So what's actually new here?

The key point is that the PX4's estimator reacts to real physical motion, not synthesized signals. In conventional HILS, IMU data is software-generated. In this system, the estimator experiences real inertial excitation. That changes what you can validate.

Now, to be upfront: this doesn't replace conventional HILS or real flight tests. The platform has a finite workspace, actuation limits, and it can't reproduce aerodynamic loads. So think of it as a complementary layer — one that gives you physical evidence that a pure software loop can't.

The Stewart platform serves as the physical link between the simulator and the flight controller — converting software attitude commands into real 6-DOF motion so the PX4 IMU actually moves. I'll show you the detailed architecture next.

## Slide 4 — System in action

[Show hardware photo and/or demo video]

Here's the system in action. You can see the Stewart platform moving to match the simulated aircraft attitude, with the PX4 sitting on top, sensing everything in real time.

## Slide 5 — How the physical closed loop works

Now let me walk you through how the closed loop actually works.

[Show Fig. 1: System architecture/data flow with numbered callouts 1–4]

At a high level, there are four steps:

1. X-Plane sends simulated vehicle state to the host over UDP.
2. The host converts that into platform pose commands.
3. The Stewart platform creates real inertial motion for PX4.
4. PX4 attitude returns to the host, where the host's PID controller computes control-surface commands and injects them into X-Plane.

Software, mechanics, physical sensing, and control I/O — all connected in one loop. Let me walk through each block.

## Slide 6 — Block 1: X-Plane simulator

[Architecture figure with Block 1 highlighted]

The loop starts here. X-Plane computes the aircraft dynamics and provides a repeatable scenario environment. Vehicle state is published to the host over UDP.

In a conventional setup, the loop would close entirely inside X-Plane. In this architecture, PX4 attitude comes back through real hardware, and the host PID converts it into control-surface commands that are injected into X-Plane.

## Slide 7 — Block 2: Python host / bridge

[Architecture figure with Block 2 highlighted]

The host is the integration hub of the entire system.

It simultaneously receives X-Plane state over UDP and PX4 attitude estimates over MAVLink/USB. It converts the simulator state into platform pose commands, and timestamps every stream with a common clock for logging and latency analysis.

Outbound, it sends pose commands to the platform over serial. PX4 attitude is used as feedback by the host PID, and the PID output is injected into X-Plane over UDP to close the loop. If you want to understand timing, interfaces, or logging, this is the block.

## Slide 8 — Block 3: Stewart platform actuation

[Architecture figure with Block 3 highlighted]

The third block is the Stewart platform — the hardware that turns software commands into real motion.

It takes the pose command from the host, applies workspace and saturation limits, and physically moves. Once done, it returns an ACK, processing time, saturation flag, and scaling factor alpha back to the host. With serial command-ACK around 22 ms and the RC servo PWM frame at 20 ms (50 Hz), the overall loop runs around 50 Hz.

After this point, what PX4 experiences is no longer synthetic — it's actual inertial motion.

## Slide 9 — Block 4: PX4 sensing and return

[Architecture figure with Block 4 highlighted]

Finally, PX4 closes the physical side of the loop.

Its onboard IMU senses the actual platform motion, and its estimator computes attitude in real time on real hardware. That estimate is sent back to the host over MAVLink/USB, where the host PID computes control-surface commands and injects them into X-Plane — completing the loop.

The validation target is not a software-emulated estimator. It's the actual PX4 estimator reacting to real motion.

## Slide 10 — What we tested and how we measured it

Now, what did we actually test?

We set the platform's neutral position at z = +20 mm — that maximizes the available rotation range.

The baseline scenario is a simple trim transition: pitch goes from 0 to +5 degrees, holds there, then comes back. Every run launches from the same X-Plane starting condition — a Honolulu PHNL 10-nautical-mile approach.

Before each run, there's a scripted grace period for the system to stabilize, and we monitor the Stewart platform's saturation flag throughout to make sure we stay within the feasible workspace. We ran 9 repetitions per condition for statistical confidence.

For each run we record X-Plane attitude, PX4 attitude, platform pose commands, command-ACK timing, and a set of post-processed metrics. We picked this scenario because it's simple, repeatable, and gives us clean numbers to work with.

Because we need to isolate system-inherent latency, bias, and mounting effects first, we start with a clean baseline under controlled conditions. Without that baseline, you can't separate system-level error from scenario-level error in later disturbance or saturation tests.

## Slide 11 — Result 1: Closed-loop tracking is achieved

Now let's look at the results. Starting with tracking.

[Show Fig. 2: pitch response overlay]

You can see PX4 pitch follows the commanded transition well. Since this trim transition is low-frequency, phase lag is not visually pronounced in this plot; we'll quantify latency on Slide 13. What's interesting here is the near-constant offset in the raw error.

But that offset isn't random instability. It's about +1.8 degrees, consistent across runs — a mounting bias. So we evaluated both the raw error and a bias-corrected version.

The bottom line: closed-loop tracking works, it's quantifiable, and the error structure makes physical sense.

## Slide 12 — Result 2: Mounting quality explains the dominant error

So if there's a bias, where does it come from? We validate the mounting-quality hypothesis with two pieces of evidence.

[Show Fig. 3: pitch error + Fig. 4: mounting bias evidence]

First evidence: when we change mounting quality, bias behavior changes. With loose mounting (adhesive tape only), bias grows over time as the FC physically shifts under repeated inertial loading. With firm mounting (structural adhesive plus mechanical tiedowns), that drift disappears.

Second evidence: in the firm-mount condition, the remaining bias stays nearly constant through the run. That means the residual error is not dynamic instability; it's a static alignment offset from mounting.

Taken together, the dominant pitch error comes from mounting quality, not the control system. Once mounting is fixed, error converges to a static bias that post-processing can automatically estimate and compensate.

## Slide 13 — Result 3: Latency characterization and system performance

Now, tracking accuracy is one thing — but how fast can this system actually respond? That's what limits fidelity in practice.

[Show Fig. 5(a): latency time series + Fig. 5(b): latency distribution]

There are four timing categories. First, X-Plane sample age: how old simulator data is when the host uses it, about 29 ms on average. Second, PX4 sample age: how old PX4 attitude data is when it reaches the host, about 19 ms. Third, serial RTT: host command to Stewart and ACK back, about 22 ms. Fourth, end-to-end delay: from X-Plane data arrival at the host to completed platform actuation ACK, about 51 ms.

This matters because this latency chain sets a practical limit on dynamic fidelity and shows why effective bandwidth should be measured explicitly, not inferred from the PWM frame or host tick.

To summarize: the host-platform update loop runs at about 50 Hz, and the representative end-to-end latency is about 51 ms. The timing spread is not negligible, so I am not claiming a measured bandwidth from these data alone; a dedicated sine-sweep test will be used to quantify effective bandwidth. At the selected neutral position, baseline commands stayed mostly within the available rotation range at z = +20 mm, and observed saturation across baseline runs remained in the 0-1% range. As we saw on the previous slide, the residual bias is largely stable and can be corrected in post-processing.

## Slide 14 — Takeaways

So to wrap up — three things I want you to take away.

First, integration. We built a physically closed-loop HILS that connects X-Plane, a Stewart platform, and PX4 into one working system.

Second, fidelity. Bias-corrected pitch tracking RMSE is about 0.46 degrees, with end-to-end latency around 51 ms.

Third, by characterizing the system — its latency, bias, and workspace limits — we established a quantitative baseline for future expansion to disturbance testing and other vehicle types.

For next steps: in the near term, time-aligned error analysis, sine-sweep-based bandwidth measurement, repeatability characterization, and stress tests near the saturation boundary. Longer term, we want to extend to other vehicle types — multirotor, spacecraft — to demonstrate the architecture's generality.

## Slide 15 — End

Thank you. I'm happy to take any questions.

## Slide 16 — Q&A backup

Use this slide only if a specific question comes up. Otherwise stay on Slide 15.

---

## Optional live delivery cues

- If time is short, skim Slides 6–9 (block details) quickly and let Slide 5 overview carry the load.
- Spend most technical depth on Slide 7 (host), 11 (tracking), and 13 (latency).
- Slide 4 (demo footage) is for audience engagement — keep it brief, 15–20 seconds.

---

## Q&A prepared answers

**"Why z=+20 mm as the neutral position?"** → Slide 16, left panel with proof figure.
The available rotation range depends on z height. At z=+20 mm, the available range in that neutral position is near its maximum; the firmware's worst-direction rotation limit is about 20° there, versus about 10° around z=0 mm. This maximizes the usable attitude envelope.

> 회전 가용 범위는 z 높이에 따라 달라집니다. z=+20 mm에서는 그 중립 위치에서의 가용 범위가 거의 최대가 되며, firmware의 최악 방향 회전 한계가 약 20° 수준입니다. 반면 z=0 mm 부근에서는 약 10° 수준으로 줄어듭니다. 그래서 자세 구현 범위를 최대화하기 위해 z=+20 mm를 선택했습니다.

**"Why not just conventional HILS?"** → Slide 3.
We add real inertial excitation that sensor synthesis cannot provide, while honestly acknowledging workspace and aerodynamic reproduction limits.

> 센서 신호 합성만으로는 줄 수 없는 실제 관성 운동을 제공하면서도, 작업 영역과 공기역학적 재현의 한계는 솔직하게 인정하고 있습니다.

**"How do you handle saturation?"** → Slides 8 + 10 + 14.
We compare the baseline commands against the available rotation range at z=+20 mm and monitor the Stewart saturation flag throughout. In the refreshed baseline metrics, observed saturation is 0-1% depending on the run, so the baseline mostly represents nominal-region behavior. Dedicated stress tests near the saturation boundary are still planned.

> baseline 명령이 z=+20 mm에서의 가용 회전 범위 안에 들어오는지 확인하고, Stewart saturation flag를 계속 모니터링합니다. 갱신된 baseline 지표에서는 run에 따라 포화 비율이 0-1% 수준으로 나타나므로, baseline은 대체로 정상 영역 거동을 보여준다고 해석할 수 있습니다. 포화 경계에 근접하는 스트레스 시험은 향후 별도로 수행할 예정입니다.

**"What about yaw?"** → Slide 10.
Yaw is in the loop, but the trim-transition scenario primarily excites pitch. Yaw tracking and bias alignment are handled in the code but not the focus of this presentation.

> Yaw도 루프에 포함되어 있지만, 트림 전환 시나리오에서는 주로 피치가 여기됩니다. Yaw 추종과 바이어스 정렬은 코드에 구현되어 있으나, 이번 발표에서는 다루지 않습니다.

**"What's the effective bandwidth of this system?"** → Slides 7 + 13.
Today I can support only the 50 Hz host tick and the measured end-to-end latency of about 51 ms. The effective attitude bandwidth will be reported from a dedicated sine-sweep measurement; placeholder wording for the final version is: "effective attitude bandwidth ~X.X Hz, with phase lag ~Y° at 1 Hz." Until that test is complete, I describe the system only as suited to low-frequency maneuvers such as trim transitions.

> 현재 제가 직접 근거로 제시할 수 있는 것은 호스트 tick 약 50 Hz와 종단간 지연 약 51 ms뿐입니다. 유효 attitude bandwidth는 전용 sine-sweep 측정으로 제시할 예정이며, 최종 문구의 placeholder는 예를 들어 "effective attitude bandwidth ~X.X Hz, with phase lag ~Y° at 1 Hz" 같은 형태가 될 것입니다. 그 전까지는 이 시스템이 트림 전환 같은 저주파 기동에 적합하다고만 설명하는 것이 타당합니다.

**"The platform is open-loop — how do you trust pose accuracy?"** → Slide 3.
A design goal was to keep the Stewart platform as affordable as possible, which led to using inexpensive RC aircraft-grade hobby servos without encoder feedback — hence the open-loop actuation. Despite this, we measured the platform's static pose accuracy with a digital inclinometer: approximately 2.1° in roll and 1.5° in pitch. So we have a practical bound on platform-level error even without continuous feedback. The PX4 IMU remains the only closed-loop physical measurement; we measure the output (attitude) but not the intermediate stage (platform pose). Adding encoders is a possible future improvement.

> 설계 목표 중 하나가 스튜어트 플랫폼을 최대한 저비용으로 구현하는 것이었습니다. 이 때문에 인코더 피드백이 없는 저가 RC용 취미 서보를 선택하게 되었고, 결과적으로 개루프 구동 방식이 된 것입니다. 그럼에도 디지털 경사계로 플랫폼의 정적 자세 정확도를 측정한 결과 롤 약 2.1°, 피치 약 1.5°였습니다. 연속 피드백 없이도 플랫폼 수준 오차의 실질적 상한을 확인한 셈입니다. PX4 IMU가 루프 안에서 유일한 폐루프 물리 측정 수단이며, 최종 출력인 자세는 측정하지만 중간 단계인 플랫폼 포즈는 측정하지 못합니다. 인코더 추가는 향후 개선 방안으로 고려하고 있습니다.

**"What is the mounting bias physically? Can you eliminate it?"** → Slide 12.
The ~1.8° bias comes from the physical alignment of the FC on the platform top plate. It is consistent within a run (std ~0.2°) and can be calibrated out in post-processing, which we do. The physical mount could also be shimmed or re-aligned, but for HILS validation purposes, software correction is sufficient as long as the bias is stable.

> 약 1.8°의 바이어스는 플랫폼 상판 위에 FC를 장착할 때의 물리적 정렬 오차에서 비롯됩니다. 실험 내에서는 일관되게 나타나고(표준편차 약 0.2°), 후처리로 보정할 수 있으며 실제로 보정하고 있습니다. 마운트를 물리적으로 조정할 수도 있지만, HILS 검증 목적으로는 바이어스가 안정적인 한 소프트웨어 보정으로 충분합니다.

**"How repeatable are the results across runs?"** → Slides 12 + 14.
We have 5 baseline runs (#5–#9). Within each run, bias std is ~0.2°. Across runs, the bias estimate varies by up to 0.8°, likely due to thermal drift and minor re-seating between runs. Full repeatability characterization is listed as a next step.

> 기본 실험을 5회(#5~#9) 수행했습니다. 실험 내 바이어스의 표준편차는 약 0.2°이고, 실험 간에는 최대 0.8°까지 차이가 나는데, 이는 열적 드리프트와 미세한 재장착 차이 때문으로 보입니다. 체계적인 반복성 분석은 향후 과제입니다.

**"Can this extend to quadrotors or other vehicles?"** → Slide 14.
In principle, yes — the architecture is not tied to fixed-wing only. But this talk demonstrates fixed-wing as the first case, and cross-vehicle validation should be treated as future work until we actually run those vehicles on the platform.

> 원리적으로는 가능합니다. 시스템 구조가 고정익에만 묶여 있지는 않습니다. 다만 이번 발표에서는 고정익을 첫 번째 실증 사례로 다루고 있으며, 다른 기체로의 확장 검증은 실제로 수행한 뒤에 주장하는 것이 맞습니다. 따라서 현재로서는 향후 과제로 보는 것이 적절합니다.
