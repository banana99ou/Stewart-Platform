# Conference Presentation Script

Korean spoken script first (for reading aloud), English reference below (for LLM context).
Slide numbers match the current deck (12 slides).

---

# 발표 대본 (Korean Spoken Script)

## 슬라이드 1 — 제목

안녕하세요, 국민대학교 미래모빌리티제어연구실 석사과정 정현용입니다. 오늘은 스튜어트 플랫폼을 이용한 물리적 폐루프 HILS 시험 환경 구축에 대해 말씀드리겠습니다.

간단히 말씀드리면, X-Plane 시뮬레이터, 실물 스튜어트 플랫폼, 실물 PX4 비행 제어기를 하나의 루프로 연결했습니다. 시뮬레이터가 플랫폼을 물리적으로 움직이고, PX4가 그 움직임을 감지해서, 그 측정값이 다시 시뮬레이션으로 돌아가는 구조입니다.

연구 동기부터 시스템 구성, 실험 내용, 측정 결과, 향후 계획까지 순서대로 말씀드리겠습니다.

`>> NEXT SLIDE`

## 슬라이드 2 — 왜 물리적 운동을 HILS에 추가하는가?

그러면 왜 HILS에 물리적 운동을 추가해야 할까요?

기존 HILS도 강력합니다. 하지만 대부분의 구성에서 비행 제어기는 합성된 센서 데이터를 받고, 자세는 화면이나 로그 파일로 확인하죠. IMU가 실제로 움직이지는 않습니다.

저희가 원한 것은 물리적 현실감입니다. 비행 제어기의 IMU가 실제 관성 운동을 겪도록 하자는 것이 출발점이었습니다.

그리고 이건 고정익에만 해당하는 이야기가 아닙니다. 쿼드콥터, 인공위성, 유도탄 — 자세 동역학이 중요한 모든 비행체에 같은 개념을 적용할 수 있습니다.

그래서 연구 질문은 이렇게 됩니다. 안정적이고 반복 가능한 물리 폐루프 시험 환경을 만들 수 있는가, 그리고 추종 성능을 실제로 정량화할 수 있는가.

`>> NEXT SLIDE`

## 슬라이드 3 — 무엇이 새로운가?

그러면 구체적으로 뭐가 새로운 걸까요?

차이점은 루프 안에 하드웨어가 있다는 것 자체가 아닙니다. 하드웨어가 실제로 물리량을 센싱한다는 점입니다. PX4의 추정기가 주입된 신호가 아니라 실제 플랫폼 움직임에 반응합니다. 그래서 플랫폼 한계 내에서 더 현실적인 폐루프 거동이 나옵니다.

다만 미리 말씀드리면, 이 시스템이 기존 HILS나 실제 비행 시험을 대체하지는 않습니다. 플랫폼의 작업 영역은 유한하고, 구동 한계가 있고, 공기역학적 하중도 재현하지 못합니다. 보완적 검증 수단으로 보시면 됩니다 — 다만 순수 소프트웨어 루프로는 얻을 수 없는 물리적 근거를 추가한다는 점이 핵심입니다.

그리고 한 가지 더 중요한 점이 있습니다. 이 구조는 어떤 기체를 시험하는지에 종속되지 않습니다. 스튜어트 플랫폼은 범용 6자유도 모션 장치니까요. 시뮬레이터 모델과 제어기 튜닝만 바꾸면 멀티로터든 위성 자세 시뮬레이터든 검증할 수 있습니다.

`>> NEXT SLIDE`

## 슬라이드 4 — 시스템 구동 모습

[사진 및/또는 데모 영상]

실제 시스템이 작동하는 모습입니다. 스튜어트 플랫폼이 시뮬레이션 자세에 맞춰 움직이고, 위에 탑재된 PX4가 그 움직임을 실시간으로 감지하고 있습니다.

`>> NEXT SLIDE`

## 슬라이드 5 — 물리적 폐루프 동작 원리

이제 폐루프가 실제로 어떻게 동작하는지 보겠습니다.

[그림 1: 시스템 구조/데이터 흐름도]

네 단계로 이루어집니다.

1. X-Plane이 UDP로 비행체 상태를 Python 호스트에 보냅니다.
2. 호스트가 이를 스튜어트 플랫폼의 자세 명령으로 변환합니다.
3. 플랫폼 위의 PX4가 자체 IMU로 물리적 움직임을 감지합니다.
4. PX4의 자세 추정값이 호스트를 거쳐 X-Plane에 제어 입력으로 들어갑니다.

이렇게 소프트웨어, 기구부, 물리 센싱, 제어 I/O가 모두 하나의 완전한 루프로 연결됩니다.

`>> NEXT SLIDE`

## 슬라이드 6 — 실험 설계 및 측정 방법

자, 그러면 실제로 무엇을 실험했는지 보겠습니다.

플랫폼 중립 위치는 z = +20 mm으로 설정했습니다. 회전 가용 범위가 최대가 되는 높이입니다.

기본 시나리오는 트림 전환입니다. 피치를 0도에서 +5도로 올렸다가, 유지한 뒤, 다시 돌아옵니다. 모든 실험은 동일한 X-Plane 초기 조건인 호놀룰루 PHNL 10해리 어프로치에서 시작합니다.

각 실험 전에 스크립트로 설정한 유예 기간 동안 시스템을 안정화하고, 실험 내내 스튜어트 플랫폼의 포화 플래그를 감시하여 구동 가능 영역 안에 있는지 확인합니다. 조건별로 9회 반복하여 통계적 유의미성을 확보했습니다.

각 실험에서 X-Plane 자세, PX4 자세, 플랫폼 자세 명령, 명령-ACK 타이밍, 후처리 메트릭을 기록합니다. 단순하고 반복 가능하며, 정량 분석에 바로 쓸 수 있기 때문에 이 시나리오를 선택했습니다.

`>> NEXT SLIDE`

## 슬라이드 7 — 결과 1: 폐루프 추종 달성

이제 결과를 보겠습니다. 먼저 추종 성능입니다.

[그림 2: 피치 응답 오버레이]

보시면 PX4 피치가 명령 전환을 따라가고 있습니다. 예상대로 시간 지연이 보이는데, 정확한 수치는 다음 결과에서 분석하겠습니다. 주목할 점은 보정 전 오차에 거의 일정한 오프셋이 포함되어 있다는 것입니다.

이 오프셋은 무작위적 불안정이 아닙니다. 약 +1.8도로 실험 간에 일관되게 나타나며, 장착 정렬에서 오는 바이어스로 판단됩니다. 그래서 보정 전 오차와 바이어스 보정 후 오차를 둘 다 평가했습니다.

결론적으로, 폐루프 추종이 작동하고, 수치로 평가할 수 있으며, 오차 구조가 물리적으로 설명이 됩니다.

`>> NEXT SLIDE`

## 슬라이드 8 — 결과 2: 장착 품질이 주된 오차의 원인

바이어스가 있다면, 어디서 오는 걸까요? 이 슬라이드가 그 답을 보여줍니다.

[그림 3: 피치 오차 + 그림 4: 장착 바이어스 근거]

두 가지 장착 조건을 비교했습니다. 접착 테이프만으로 가볍게 고정한 느슨한 장착과, 구조용 접착제에 기계적 결속을 더해 상판에 단단히 고정한 강성 장착입니다.

결과가 분명합니다. 느슨한 장착에서는 오차가 시간이 지나면서 점점 커집니다 — 반복적인 관성 하중에 기체가 서서히 밀리는 겁니다. 단단한 장착에서는 오차가 전 구간에 걸쳐 평탄하고 단조롭게 유지됩니다.

즉, 주된 피치 오차는 제어 시스템에서 나오는 게 아닙니다. 장착 품질의 문제입니다. 장착을 잡으면 오차 대부분이 사라지고, 남는 것은 시험 기체를 얼마나 단단히 고정했느냐에 따른 거의 일정한 바이어스입니다.

`>> NEXT SLIDE`

## 슬라이드 9 — 결과 3: 지연 특성화 및 시스템 성능

추종 정확도는 봤는데, 그러면 이 시스템이 얼마나 빠르게 반응할 수 있을까요? 실질적으로 충실도를 제한하는 건 바로 이 부분입니다.

[그림 5(a): 지연 시계열 + 그림 5(b): 지연 분포]

먼저 타이밍 용어를 간단히 짚겠습니다. Sample age는 데이터를 실제로 쓸 때 그 데이터가 얼마나 오래된 것인지입니다. Serial RTT는 스튜어트 플랫폼에 명령을 보내고 응답을 받기까지의 왕복 시간입니다. 종단간 지연은 시뮬레이터 데이터 도착부터 플랫폼이 동작을 완료하기까지의 전체 시간입니다.

수치를 보면, serial RTT 약 22 ms, PX4 수신-ACK 간 약 41 ms, X-Plane 수신-ACK 전체 루프가 약 51 ms입니다.

이게 중요한 이유는, 지연이 추종에서 보이는 위상 지연의 직접적인 원인이고, 루프가 동적 명령을 얼마나 잘 따라갈 수 있는지의 상한을 정하기 때문입니다.

전체 시스템 그림을 보면, 약 50 Hz로 동작하고, 추종 위상 지연은 측정된 종단간 지연과 일치합니다. 지터는 낮아서 타이밍 분포가 좁게 모여 있고요. ±5도가 가용 범위 20도 안에 충분히 들어가기 때문에 어떤 실험에서도 포화는 없었습니다. 그리고 앞 슬라이드에서 봤듯이 바이어스는 안정적이고 보정 가능하며, 실험 내 드리프트는 무시할 수 있는 수준이었습니다.

`>> NEXT SLIDE`

## 슬라이드 10 — 핵심 정리

정리하겠습니다. 네 가지 핵심 사항입니다.

첫째, 통합. X-Plane, 스튜어트 플랫폼, PX4를 하나의 물리적 폐루프 HILS로 구현했습니다.

둘째, 충실도. 폐루프 추종이 동작하고, 수치로 평가할 수 있습니다.

셋째, 투명성. 바이어스와 지연을 숨기지 않고 측정해서, 이 플랫폼이 실제로 무엇을 할 수 있는지 정의했습니다.

넷째, 범용성. 이 구조는 특정 기체에 묶이지 않습니다. 쿼드콥터, 인공위성, 유도탄 — 자세 동역학이 중요한 모든 비행체에 같은 시험 환경을 쓸 수 있습니다.

향후 계획으로는, 단기적으로 시간 정렬 오차 분석, 반복성 특성화, 포화 경계 부근 스트레스 시험을 예정하고 있고요. 장기적으로는 다양한 기체를 같은 하드웨어에서 검증해서 범용성을 실증하려고 합니다.

`>> NEXT SLIDE`

## 슬라이드 11 — 마무리

감사합니다. 질문 있으시면 말씀해 주세요.

`>> BACKUP SLIDE (if needed)`

## 슬라이드 12 — Q&A 백업

특정 질문이 나올 때만 이 슬라이드를 사용합니다. 그 외에는 슬라이드 11에 머물러 주세요.

---

## 발표 운용 참고사항

- 시간 부족 시 슬라이드 5, 6을 합쳐 1분 내로 압축.
- 기술적 심화는 슬라이드 5, 7, 9에 집중.
- 슬라이드 4 (데모 영상)는 청중 몰입용 — 15~20초 이내.

---
---

# English Reference Script

The English version below is the reference/source text for the Korean spoken script above.
It also contains Q&A prepared answers for anticipated questions.

---

## Slide 1 — Title / One-line contribution

Hello, I'm Hyeon-yong Jeong. Today I'll be talking about our work building a physically closed-loop HILS testbed using a Stewart platform.

In short — we connected X-Plane, a real Stewart platform, and a real PX4 flight controller into one loop. The simulator drives physical motion, the PX4 senses it, and that measurement feeds right back into the simulation.

I'll walk you through the motivation, how we built it, what we tested, what we measured, and where we go next.

## Slide 2 — Why add physical motion to HILS?

So why add physical motion to HILS?

Conventional HILS is powerful. But in most setups, the flight controller gets synthesized sensor data. You check attitude on a screen or in a log file. The IMU never actually moves.

What we wanted was physical realism — to make the FC's IMU experience real inertial motion, not just numbers fed through a wire.

And this isn't limited to fixed-wing aircraft. Quadcopters, satellites, missiles — anything where attitude dynamics matter — could use the same concept.

So the research question becomes: can we build a stable, repeatable, physically closed-loop testbed and actually quantify how well it tracks?

## Slide 3 — What is new here?

So what's actually new here?

The difference isn't just that there's hardware in the loop. It's that the hardware is physically sensing. The PX4's estimator reacts to real platform motion, not injected signals. That gives you a more realistic closed-loop behavior — within the platform's limits, of course.

Now, to be upfront: this doesn't replace conventional HILS or real flight tests. The platform has a finite workspace, actuation limits, and it can't reproduce aerodynamic loads. So think of it as a complementary layer — one that gives you physical evidence that a pure software loop can't.

One more important point: this architecture doesn't care what vehicle you're testing. The Stewart platform is a general-purpose 6-DOF motion device. Swap the simulator model and retune the controller, and you can validate anything from a multirotor to a spacecraft attitude simulator.

## Slide 4 — System in action

[Show hardware photo and/or demo video]

Here's the system in action. You can see the Stewart platform moving to match the simulated aircraft attitude, with the PX4 sitting on top, sensing everything in real time.

## Slide 5 — How the physical closed loop works

Now let me walk you through how the closed loop actually works.

[Show Fig. 1: System architecture/data flow with numbered callouts]

There are four steps:

1. X-Plane sends the vehicle state to our Python host over UDP.
2. The host converts that into pose commands for the Stewart platform.
3. The PX4, mounted on the platform, senses the physical motion through its own IMU.
4. PX4's attitude estimate goes back to the host, which injects it as control input into X-Plane.

So you have a complete loop — software, mechanics, physical sensing, and control I/O, all connected.

## Slide 6 — What we tested and how we measured it

Now, what did we actually test?

We set the platform's neutral position at z = +20 mm — that maximizes the available rotation range.

The baseline scenario is a simple trim transition: pitch goes from 0 to +5 degrees, holds there, then comes back. Every run launches from the same X-Plane starting condition — a Honolulu PHNL 10-nautical-mile approach.

Before each run, there's a scripted grace period for the system to stabilize, and we monitor the Stewart platform's saturation flag throughout to make sure we stay within the feasible workspace. We ran 9 repetitions per condition for statistical confidence.

For each run we record X-Plane attitude, PX4 attitude, platform pose commands, command-ACK timing, and a set of post-processed metrics. We picked this scenario because it's simple, repeatable, and gives us clean numbers to work with.

## Slide 7 — Result 1: Closed-loop tracking is achieved

Now let's look at the results. Starting with tracking.

[Show Fig. 2: pitch response overlay]

You can see here that PX4 pitch follows the commanded transition — there's a lag, which is expected, and we'll quantify it shortly. What's interesting is that the raw error includes a near-constant offset.

But that offset isn't random instability. It's about +1.8 degrees, consistent across runs, which points to a mounting bias — a physical alignment issue. So we evaluated both the raw error and a bias-corrected version.

The bottom line: closed-loop tracking works, it's quantifiable, and the error structure makes physical sense.

## Slide 8 — Result 2: Mounting quality explains the dominant error

So if there's a bias, where does it come from? That's what this slide answers.

[Show Fig. 3: pitch error + Fig. 4: mounting bias evidence]

We tested two mounting conditions: a loosely fixed setup using only adhesive tape, and a rigidly fixed setup using structural adhesive combined with mechanical tiedowns to the top plate.

The results are telling. With loose mounting, the error gradually grows over the run — the aircraft physically shifts under repeated inertial loads. With firm mounting, the error stays flat and monotonic throughout.

So the dominant pitch error isn't coming from the control system. It's a mounting-quality artifact. Fix the mount, and most of the error goes away. What remains is a near-constant bias whose size depends on how securely the test article is attached.

## Slide 9 — Result 3: Latency characterization and system performance

Now, tracking accuracy is one thing — but how fast can this system actually respond? That's what limits fidelity in practice.

[Show Fig. 5(a): latency time series + Fig. 5(b): latency distribution]

Let me quickly define the timing terms. Sample age is how stale your data is when you actually use it. Serial RTT is the round-trip time for a command to go out to the Stewart platform and get acknowledged back. And end-to-end delay is the whole chain — from simulator data arriving to the platform completing its move.

The numbers: serial RTT is about 22 milliseconds. PX4 receive-to-ACK is about 41 ms. The full X-Plane receive-to-ACK loop takes about 51 ms.

This matters because latency directly sets the phase lag you see in tracking, and it puts a hard ceiling on how well the loop can follow dynamic commands.

Looking at the overall system picture: we're running at about 50 Hz. The phase lag in tracking matches the measured end-to-end latency. Jitter is low — the timing distributions are tight. No saturation was observed in any run, since our ±5 degrees is well within the 20 degrees of available workspace. And as we saw on the previous slide, bias is stable and correctable, with negligible drift within each run.

## Slide 10 — Takeaways

So to wrap up — four things I want you to take away.

First, integration. We built a physically closed-loop HILS that connects X-Plane, a Stewart platform, and PX4 into one working system.

Second, fidelity. Closed-loop tracking works, and we can put numbers on it.

Third, transparency. We didn't hide the bias or latency — we measured them and used them to define what this platform can actually do.

And fourth, generality. This architecture isn't tied to one vehicle type. The same testbed can validate quadcopters, satellites, missiles — anything with attitude dynamics that matter.

For next steps: in the near term, time-aligned error analysis, repeatability characterization, and stress tests near the saturation boundary. Longer term, we want to demonstrate that cross-vehicle generality by running different vehicle types on the same hardware.

## Slide 11 — End

Thank you. I'm happy to take any questions.

## Slide 12 — Q&A backup

Use this slide only if a specific question comes up. Otherwise stay on Slide 11.

---

## Optional live delivery cues

- If time is short, compress Slides 5 and 6 into one minute total.
- Spend most technical depth on Slides 5, 7, and 9.
- Slide 4 (demo footage) is for audience engagement — keep it brief, 15–20 seconds.

---

## Q&A prepared answers

**"Why z=+20 mm as the neutral position?"** → Slide 12, left panel with proof figure.
The rotation range is a function of z height. At z=+20 mm we get ~20° of rotation; at z=0 mm only ~10°. This maximizes the usable attitude envelope.

> 회전 가용 범위가 z 높이에 따라 달라집니다. z=+20 mm에서는 약 20°, z=0 mm에서는 약 10°밖에 안 됩니다. 자세 구현 범위를 최대로 확보하기 위한 선택입니다.

**"Why not just conventional HILS?"** → Slide 3.
We add real inertial excitation that sensor synthesis cannot provide, while honestly acknowledging workspace and aerodynamic reproduction limits.

> 센서 신호 합성만으로는 줄 수 없는 실제 관성 운동을 제공하면서도, 작업 영역과 공기역학적 재현의 한계는 솔직하게 인정하고 있습니다.

**"How do you handle saturation?"** → Slides 6 + 9 + 10.
No saturation was observed in any baseline run. The operating range of ±5° sits well within the ~20° available workspace, so the baseline results represent purely nominal-region behavior. Stress tests approaching saturation boundaries are a planned next step.

> 어떤 기본 실험에서도 포화는 관측되지 않았습니다. ±5°의 운용 범위는 약 20° 가용 작업 공간 안에 충분히 여유가 있어, 기본 실험 결과는 순수하게 정상 작동 영역의 거동을 나타냅니다. 포화 경계에 근접하는 스트레스 시험은 향후 계획하고 있습니다.

**"What about yaw?"** → Slide 6.
Yaw is in the loop, but the trim-transition scenario primarily excites pitch. Yaw tracking and bias alignment are handled in the code but not the focus of this presentation.

> Yaw도 루프에 포함되어 있지만, 트림 전환 시나리오에서는 주로 피치가 여기됩니다. Yaw 추종과 바이어스 정렬은 코드에 구현되어 있으나, 이번 발표에서는 다루지 않습니다.

**"What's the effective bandwidth of this system?"** → Slides 5 + 9.
The host ticks at ~50 Hz. End-to-end latency from X-Plane receive to Stewart ACK is ~51 ms. This means the system is suited for low-frequency maneuvers like trim transitions; high-frequency dynamic content is filtered by the latency chain. Increasing bandwidth would require reducing serial overhead or moving to a faster transport.

> 호스트 루프가 약 50 Hz로 동작하고, X-Plane 수신부터 Stewart ACK까지 종단간 지연이 약 51 ms입니다. 따라서 트림 전환 같은 저주파 기동에 적합하며, 고주파 동역학 성분은 지연에 의해 걸러집니다. 대역폭을 높이려면 시리얼 오버헤드를 줄이거나 더 빠른 통신 수단으로 전환해야 합니다.

**"The platform is open-loop — how do you trust pose accuracy?"** → Slide 3.
A design goal was to keep the Stewart platform as affordable as possible, which led to using inexpensive RC aircraft-grade hobby servos without encoder feedback — hence the open-loop actuation. Despite this, we measured the platform's static pose accuracy with a digital inclinometer: approximately 2.1° in roll and 1.5° in pitch. So we have a practical bound on platform-level error even without continuous feedback. The PX4 IMU remains the only closed-loop physical measurement; we measure the output (attitude) but not the intermediate stage (platform pose). Adding encoders is a possible future improvement.

> 설계 목표 중 하나가 스튜어트 플랫폼을 최대한 저비용으로 구현하는 것이었습니다. 이 때문에 인코더 피드백이 없는 저가 RC용 취미 서보를 선택하게 되었고, 결과적으로 개루프 구동 방식이 된 것입니다. 그럼에도 디지털 경사계로 플랫폼의 정적 자세 정확도를 측정한 결과 롤 약 2.1°, 피치 약 1.5°였습니다. 연속 피드백 없이도 플랫폼 수준 오차의 실질적 상한을 확인한 셈입니다. PX4 IMU가 루프 안에서 유일한 폐루프 물리 측정 수단이며, 최종 출력인 자세는 측정하지만 중간 단계인 플랫폼 포즈는 측정하지 못합니다. 인코더 추가는 향후 개선 방안으로 고려하고 있습니다.

**"What is the mounting bias physically? Can you eliminate it?"** → Slide 8.
The ~1.8° bias comes from the physical alignment of the FC on the platform top plate. It is consistent within a run (std ~0.2°) and can be calibrated out in post-processing, which we do. The physical mount could also be shimmed or re-aligned, but for HILS validation purposes, software correction is sufficient as long as the bias is stable.

> 약 1.8°의 바이어스는 플랫폼 상판 위에 FC를 장착할 때의 물리적 정렬 오차에서 비롯됩니다. 실험 내에서는 일관되게 나타나고(표준편차 약 0.2°), 후처리로 보정할 수 있으며 실제로 보정하고 있습니다. 마운트를 물리적으로 조정할 수도 있지만, HILS 검증 목적으로는 바이어스가 안정적인 한 소프트웨어 보정으로 충분합니다.

**"How repeatable are the results across runs?"** → Slides 8 + 10.
We have 5 baseline runs (#5–#9). Within each run, bias std is ~0.2°. Across runs, the bias estimate varies by up to 0.8°, likely due to thermal drift and minor re-seating between runs. Full repeatability characterization is listed as a next step.

> 기본 실험을 5회(#5~#9) 수행했습니다. 실험 내 바이어스의 표준편차는 약 0.2°이고, 실험 간에는 최대 0.8°까지 차이가 나는데, 이는 열적 드리프트와 미세한 재장착 차이 때문으로 보입니다. 체계적인 반복성 분석은 향후 과제입니다.

**"Can this extend to quadrotors or other vehicles?"** → Slide 10.
The architecture is vehicle-agnostic. Only the X-Plane aircraft model and the PID tuning change. The Stewart platform is a general 6-DOF motion device. PX4 itself supports both fixed-wing and multirotor, so the same hardware setup can validate different vehicle classes.

> 시스템 구조가 특정 기체에 종속되지 않습니다. X-Plane의 기체 모델과 PID 튜닝만 바꾸면 됩니다. 스튜어트 플랫폼은 범용 6자유도 모션 장치이고, PX4 자체가 고정익과 멀티로터를 모두 지원하므로, 동일한 하드웨어 구성으로 다양한 기체를 검증할 수 있습니다.
