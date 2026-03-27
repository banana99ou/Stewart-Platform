# Conference Presentation Script

Korean spoken script first (for reading aloud), English reference below (for LLM context).
Slide numbers match the current deck (17 slides).

---

# 발표 대본 (Korean Spoken Script)

## 슬라이드 1 — 제목

안녕하세요, 국민대학교 미래모빌리티제어연구실 석사과정 정현용입니다.

대부분의 HILS에서 비행 제어기의 IMU는 실제로 움직이지 않습니다. 저희는 그것을 
바꿨습니다. X-Plane 시뮬레이터, 실물 스튜어트 플랫폼, 실물 PX4를 하나의 루프로 
연결해서, 시뮬레이션 자세가 실제 물리적 운동으로 바뀌고, PX4가 그 운동을 감지하고, 그 
측정값이 다시 시뮬레이션으로 돌아가는 구조를 만들었습니다.
---
HILS를 해보신 분들은 아시겠지만, 보통 비행 제어기의 IMU는 책상 위에 가만히 놓여 있습니다. 센서 데이터는 소프트웨어로 주입하고, 자세는 화면에서 확인하죠.

저희는 여기에 실제 물리적 운동을 넣었습니다. X-Plane, 스튜어트 플랫폼, PX4를 하나의 루프로 묶어서, 시뮬레이션 자세가 플랫폼의 실제 움직임이 되고, PX4가 그걸 감지해서 다시 시뮬레이션으로 돌아가는 구조입니다. 플랫폼이 기울면 위에 탑재된 비행 제어기의 조종면이 실제로 반응하는 걸 눈으로 볼 수 있습니다.

오늘은 이 시스템의 구조, 기준선 실험 결과, 그리고 정량적으로 확인한 가능성과 한계를 보여드리겠습니다.

`>> NEXT SLIDE`

## 슬라이드 2 — 왜 물리적 운동을 HILS에 추가하는가?

그러면 왜 HILS에 물리적 운동을 추가해야 할까요?

기존 HILS에서도 충분히 많은 걸 할 수 있습니다. 다만 대부분의 구성에서 비행 제어기는 합성된 센서 데이터를 받고, 자세는 화면이나 로그에서 확인하죠. IMU는 실제로 움직이지 않습니다.

저희 목표는 단순했습니다. 비행 제어기가 실제 움직임을 겪게 하자. IMU가 관성 운동을 경험하고, 그 결과 조종면이 반응하는 걸 눈으로 보자 — 이것이 출발점이었습니다.

고정익으로 먼저 실증했고, 원리적으로는 다른 기체에도 적용 가능합니다.

오늘은 이 물리 폐루프 시험 환경이 안정적으로 동작하고, 추종 성능을 정량화할 수 있다는 것을 실험 데이터로 보여드리겠습니다.

`>> NEXT SLIDE`

## 슬라이드 3 — 무엇이 새로운가?

그러면 구체적으로 뭐가 새로운 걸까요?

핵심은 PX4의 추정기가 합성 신호가 아니라 실제 물리적 움직임에 반응한다는 점입니다. 기존 HILS에서는 IMU 데이터가 소프트웨어로 생성되지만, 이 시스템에서는 실제 관성 운동을 추정기가 직접 경험합니다.

이를 위해 스튜어트 플랫폼이 시뮬레이터와 비행 제어기 사이의 물리적 연결 고리 역할을 합니다. 소프트웨어 자세 명령을 실제 6자유도 운동으로 바꿔서, PX4 IMU가 실제로 움직이게 합니다.

이 물리적 자극이 추가됨으로써 무엇이 드러나는지, 실제 데이터로 보여드리겠습니다. 먼저 시스템이 작동하는 모습부터 보겠습니다.

`>> NEXT SLIDE`

## 슬라이드 4 — 시스템 구동 모습

[데모 영상 재생]

지금 보시는 것은 기준선 실험인 트림 전환 시나리오입니다. X-Plane이 피치 0도에서 +5도로 전환하는 명령을 내리면, 스튜어트 플랫폼이 그에 맞춰 기울어지고, 위에 탑재된 PX4가 이 움직임을 실시간으로 감지합니다.

잠시 후 결과 슬라이드에서 이 동작에 대응하는 추종 데이터를 보여드리겠습니다. 먼저 시스템 구조를 살펴보겠습니다.

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

X-Plane이 동역학을 연산하고, UDP로 기체 상태를 호스트에 전송합니다. 기존에는 이 안에서 루프가 닫혔지만, 이 구조에서는 PX4 자세가 물리 경로를 거쳐 되돌아옵니다.

`>> NEXT SLIDE`

## 슬라이드 7 — 블록 2: Python 호스트

[구조도에서 2번 블록 강조]

호스트는 통합 허브입니다. X-Plane 상태와 PX4 추정값을 동시에 수신하고, 플랫폼 명령 변환, 공통 타임스탬프 로깅, PID 제어 출력까지 모두 여기서 처리합니다.

`>> NEXT SLIDE`

## 슬라이드 8 — 블록 3: Stewart 플랫폼 구동

[구조도에서 3번 블록 강조]

스튜어트 플랫폼은 호스트의 pose 명령을 받아 실제 물리적 모션으로 변환합니다. 비용을 최소화하기 위해 50 Hz RC용 취미 서보를 사용했고, 시리얼 명령-ACK 왕복 시간은 약 22 ms입니다. 이 블록 이후부터 PX4가 경험하는 것은 실제 관성 운동입니다.

`>> NEXT SLIDE`

## 슬라이드 9 — 블록 4: PX4 센싱 및 귀환

[구조도에서 4번 블록 강조]

PX4의 IMU가 플랫폼의 실제 운동을 감지하고, 추정기가 자세를 산출합니다. 이 추정값이 호스트로 돌아와 PID 제어기를 거쳐 X-Plane에 주입되면 루프가 완성됩니다.

`>> NEXT SLIDE`

## 슬라이드 10 — 실험 설계 및 측정 방법

자, 그러면 실제로 무엇을 실험했는지 보겠습니다.

플랫폼 중립 위치는 z = +20 mm으로 설정했습니다. 회전 가용 범위가 최대가 되는 높이입니다.

기본 시나리오는 트림 전환입니다. 피치를 0도에서 +5도로 올렸다가, 유지한 뒤, 다시 돌아옵니다. 모든 실험은 동일한 X-Plane 초기 조건인 호놀룰루 PHNL 10해리 어프로치에서 시작합니다.

각 실험은 포화 플래그를 감시하면서 9회 반복하여 통계적 유의미성을 확보했습니다. 시스템 자체의 지연, 바이어스, 장착 효과를 먼저 분리해야 하므로, 이처럼 제어된 조건의 깨끗한 기준선 시나리오에서 출발합니다.

`>> NEXT SLIDE`

## 슬라이드 11 — 결과 1: 폐루프 추종 달성

이제 결과를 보겠습니다.

[그림 2: 피치 응답 오버레이]

주황색 선이 PX4 피치, 파란 점선이 X-Plane 명령 피치입니다. 전체적으로 안정적으로 추종하고 있는데, 약 1.8도 정도 일정하게 벌어져 있는 걸 볼 수 있습니다. 이건 장착 바이어스로 추정됩니다. 이 바이어스만큼 보정한 것이 파란 실선인데, 보정 후에는 추종이 훨씬 가까워집니다.

핵심은, 폐루프 추종이 작동하고, 지배적 오차가 물리적으로 설명된다는 점입니다.

`>> NEXT SLIDE`

## 슬라이드 12 — 결과 2: 장착 품질이 주된 오차의 원인

바이어스가 있다면, 어디서 오는 걸까요? 장착 품질이 원인이라는 가설을 두 가지 근거로 확인했습니다.

[그림 3: 피치 오차 + 그림 4: 장착 바이어스 근거]

첫 번째 근거. 장착 조건을 바꾸면 바이어스의 거동이 달라집니다. 느슨한 장착 — 접착 테이프만 사용한 조건에서는 관성 하중에 의해 FC가 서서히 밀리면서 바이어스가 시간에 따라 증가합니다. 단단한 장착 — 접착제에 기계적 결속을 추가한 조건으로 바꾸면 이 드리프트가 사라집니다.

두 번째 근거. 단단한 장착 조건에서 남는 바이어스는 실험 내내 거의 일정합니다. 이는 남은 오차가 동적 불안정이 아니라, 장착 시 생긴 고정된 정렬 오차라는 뜻입니다.

두 근거를 종합하면, 주된 피치 오차의 원인은 제어 시스템이 아니라 장착 품질입니다. 장착을 잡으면 오차가 보정 가능한 정적 바이어스로 수렴하며, 후처리 스크립트가 이 바이어스를 자동으로 추정하고 보정합니다.

`>> NEXT SLIDE`

## 슬라이드 13 — 결과 3: 지연 특성화 및 시스템 성능

추종이 되는 것은 확인했는데, 그러면 이 시스템이 얼마나 빠르게 반응할 수 있을까요? 이 슬라이드가 그 질문에 답합니다.

[그림 5(a): 지연 시계열 + 그림 5(b): 지연 분포]

지연을 네 구간으로 분해했습니다. X-Plane 샘플 에이지는 시뮬레이터 데이터가 호스트에서 사용될 때까지의 경과 시간, PX4 샘플 에이지는 PX4 자세 추정값이 호스트에 도달할 때까지의 경과 시간, 시리얼 RTT는 호스트에서 플랫폼으로 명령을 보내고 ACK를 받기까지의 왕복 시간, 종단간 지연은 X-Plane 데이터 수신부터 플랫폼 구동 완료까지의 총 시간입니다.

왼쪽 시계열에서 각각 파란색, 주황색, 초록색, 빨간색으로 표시했고, 실선이 구간 평균, 음영이 최소-최대 범위입니다. 빨간 종단간 선이 약 40에서 60 ms 대에서 움직입니다.

오른쪽 박스 플롯은 baseline 5회 전체의 분포입니다. 종단간 중앙값이 약 48 ms이고, X-Plane 에이지 약 27 ms, PX4 에이지 약 19 ms, 시리얼 RTT 약 22 ms가 겹쳐 누적됩니다.

이렇게 누적된 지연이 시스템이 얼마나 빠른 움직임까지 재현할 수 있는지를 결정합니다. 분포에서 보시듯 지터 폭이 상당하기 때문에, 현재 데이터만으로 유효 bandwidth를 확정하기는 어렵습니다. 이 부분은 향후 sine-sweep 측정으로 정량화할 계획입니다.

`>> NEXT SLIDE`

## 슬라이드 14 — 핵심 정리

정리하겠습니다. 세 가지 핵심 사항입니다.

첫째, 통합. X-Plane, 스튜어트 플랫폼, PX4를 하나의 물리적 폐루프 HILS로 구현했습니다.

둘째, 충실도. 바이어스 보정 후 피치 추종 오차 RMSE가 약 0.46도이고 종단간 지연은 약 51 ms로, 폐루프 추종 성능을 정량적으로 제시할 수 있습니다.

셋째, 시스템 특성화를 통해 향후 확장의 기반을 확보했습니다. 지연, 바이어스, 작업 영역의 정량적 한계를 확인함으로써, 교란 실험이나 다른 기체 유형으로의 확장을 위한 기준선을 마련했습니다.

다만 이 시스템이 기존 소프트웨어 HILS나 실제 비행 시험을 대체하는 것은 아닙니다. 플랫폼의 작업 영역은 유한하고, 공기역학적 하중은 재현하지 못하며, 유효 bandwidth도 아직 측정이 필요합니다. 이 시스템의 위치는 소프트웨어 HILS와 비행 시험 사이의 보완적 검증 수단입니다.

향후 계획으로는, 단기적으로 시간 정렬 오차 분석, sine-sweep 기반 bandwidth 측정, 반복성 특성화, 포화 경계 부근 스트레스 시험을 예정하고 있습니다. 장기적으로는 멀티로터 등 다른 기체로 확장하여 아키텍처의 범용성을 실증하는 것이 목표입니다.

`>> NEXT SLIDE`

## 슬라이드 15 — 마무리

감사합니다. 질문 있으시면 말씀해 주세요.

`>> BACKUP SLIDE (if needed)`

## Q&A 답변

각 답변에 프레젠테이션 이동 지시가 포함되어 있습니다.

**"왜 z=+20 mm를 중립 위치로?"**
[프레젠테이션: 슬라이드 16으로 이동 — 증명 그림 있음]
회전 가용 범위는 z 높이에 따라 달라집니다. z=+20 mm에서는 그 중립 위치에서의 가용 범위가 거의 최대가 되며, firmware의 최악 방향 회전 한계가 약 20° 수준입니다. 반면 z=0 mm 부근에서는 약 10° 수준으로 줄어듭니다. 그래서 자세 구현 범위를 최대화하기 위해 z=+20 mm를 선택했습니다.

**"왜 기존 HILS가 아니라 이걸?"**
[프레젠테이션: 슬라이드 3으로 이동 — 핵심 차별점 슬라이드]
센서 신호 합성만으로는 줄 수 없는 실제 관성 운동을 제공하면서도, 작업 영역과 공기역학적 재현의 한계는 솔직하게 인정하고 있습니다.

**"포화는 어떻게 처리?"**
[프레젠테이션: 슬라이드 10으로 이동 — 실험 설정 슬라이드]
baseline 명령이 z=+20 mm에서의 가용 회전 범위 안에 들어오는지 확인하고, Stewart saturation flag를 계속 모니터링합니다. 갱신된 baseline 지표에서는 run에 따라 포화 비율이 0-1% 수준으로 나타나므로, baseline은 대체로 정상 영역 거동을 보여준다고 해석할 수 있습니다. 포화 경계에 근접하는 스트레스 시험은 향후 별도로 수행할 예정입니다.

**"Yaw는?"**
[프레젠테이션: 슬라이드 15 유지 — 구두 답변]
Yaw도 루프에 포함되어 있지만, 트림 전환 시나리오에서는 주로 피치가 여기됩니다. Yaw 추종과 바이어스 정렬은 코드에 구현되어 있으나, 이번 발표에서는 다루지 않습니다.

**"유효 bandwidth는?"**
[프레젠테이션: 슬라이드 13으로 이동 — 지연 데이터 보여주며 답변]
현재 근거로 제시할 수 있는 것은 종단간 지연 약 48 ms뿐입니다. 유효 attitude bandwidth는 전용 sine-sweep 측정으로 제시할 예정입니다. 그 전까지는 이 시스템이 트림 전환 같은 저주파 기동에 적합하다고만 설명하는 것이 타당합니다.

**"플랫폼이 개루프인데 포즈 정확도는?"**
[프레젠테이션: 슬라이드 15 유지 — 구두 답변]
설계 목표 중 하나가 스튜어트 플랫폼을 최대한 저비용으로 구현하는 것이었습니다. 이 때문에 인코더 피드백이 없는 저가 RC용 취미 서보를 선택하게 되었고, 결과적으로 개루프 구동 방식이 된 것입니다. 그럼에도 디지털 경사계로 플랫폼의 정적 자세 정확도를 측정한 결과 롤 약 0.1°, 피치 약 0.5°였습니다. 연속 피드백 없이도 플랫폼 수준 오차의 실질적 상한을 확인한 셈입니다. PX4 IMU가 루프 안에서 유일한 폐루프 물리 측정 수단이며, 최종 출력인 자세는 측정하지만 중간 단계인 플랫폼 포즈는 측정하지 못합니다. 인코더 추가는 향후 개선 방안으로 고려하고 있습니다.

**"장착 바이어스가 물리적으로 뭔가? 제거 가능?"**
[프레젠테이션: 슬라이드 12로 이동 — Fig 3/4 보여주며 답변]
약 1.8°의 바이어스는 플랫폼 상판 위에 FC를 장착할 때의 물리적 정렬 오차에서 비롯됩니다. 실험 내에서는 일관되게 나타나고(표준편차 약 0.2°), 후처리로 보정할 수 있으며 실제로 보정하고 있습니다. 마운트를 물리적으로 조정할 수도 있지만, HILS 검증 목적으로는 바이어스가 안정적인 한 소프트웨어 보정으로 충분합니다.

**"실험 간 반복성은?"**
[프레젠테이션: 슬라이드 12로 이동 — Fig 4 보여주며 답변]
기본 실험을 5회(#5~#9) 수행했습니다. 실험 내 바이어스의 표준편차는 약 0.2°이고, 실험 간에는 최대 0.8°까지 차이가 나는데, 이는 열적 드리프트와 미세한 재장착 차이 때문으로 보입니다. 체계적인 반복성 분석은 향후 과제입니다.

**"쿼드로터 등 다른 기체로 확장 가능?"**
[프레젠테이션: 슬라이드 14로 이동 — 향후 계획 보여주며 답변]
원리적으로는 가능합니다. 시스템 구조가 고정익에만 묶여 있지는 않습니다. 다만 이번 발표에서는 고정익을 첫 번째 실증 사례로 다루고 있으며, 다른 기체로의 확장 검증은 실제로 수행한 뒤에 주장하는 것이 맞습니다. 따라서 현재로서는 향후 과제로 보는 것이 적절합니다.

**"FC 장착 모습을 보여줄 수 있나?"**
[프레젠테이션: 슬라이드 12로 이동 — 바이어스 데이터로 리다이렉트]
장착 조건 비교 자체가 과학적 근거입니다. 장착 품질을 바꾸면 바이어스 거동이 예측 가능하게 변합니다. 그림 3, 4가 이를 보여줍니다: 느슨한 장착에서는 점진적 드리프트, 단단한 장착에서는 안정된 정적 오프셋만 남습니다. 물리적 장착은 시스템의 저비용 설계 철학에 맞춰 의도적으로 단순하게 유지했습니다. 정량적 바이어스 데이터가 장착 사진보다 더 유의미한 정보를 제공합니다.

---

## 발표 운용 참고사항

- 슬라이드 6~9(블록 상세)는 슬라이드당 15~20초로 빠르게 넘김. 시각 자료가 세부를 전달하고, 구두 설명은 핵심 역할만 짚음.
- 기술적 심화는 슬라이드 11(추종), 12(바이어스), 13(지연)에 집중.
- 슬라이드 4 (데모 영상)에서 트림 전환 시나리오를 구두로 연결 — 15~20초 이내, "이 동작의 데이터를 결과에서 보겠습니다"로 전환.

---

# English Reference Script

The English version below is the reference/source text for the Korean spoken script above.
It also contains Q&A prepared answers for anticipated questions.

---

## Slide 1 — Title / One-line contribution

Hello, I'm Hyeon-yong Jeong from Kookmin University.

In most HILS setups, the flight controller's IMU never actually moves. We changed that. We connected X-Plane, a real Stewart platform, and a real PX4 flight controller into one loop — where simulated attitude becomes real physical motion, PX4 senses it, and that measurement feeds back into the simulation.

The core question is simple: how does the PX4 estimator respond when driven by real inertial motion?

Today I'll show you the system architecture, baseline experimental results, and the quantified capabilities and limits of this approach.

## Slide 2 — Why add physical motion to HILS?

So why add physical motion to HILS?

Conventional HILS is powerful. But in most setups, the flight controller gets synthesized sensor data. You check attitude on a screen or in a log file. The IMU never actually moves.

What we wanted was physical realism — to make the FC's IMU experience real inertial motion and to see the hardware physically responding, not just numbers on a screen.

This approach is not limited to fixed-wing aircraft — any vehicle where attitude dynamics matter could use the same principle. We chose fixed-wing as the first demonstration case because X-Plane supports fixed-wing and helicopters, and among those, a fixed-wing trim transition is the simplest repeatable baseline for isolating system characteristics.

In this talk, we show with experimental data that this physically closed-loop testbed works stably and that tracking performance is quantifiable.

## Slide 3 — What is new here?

So what's actually new here?

The key point is that the PX4's estimator reacts to real physical motion, not synthesized signals. In conventional HILS, IMU data is software-generated. In this system, the estimator experiences real inertial excitation.

The Stewart platform serves as the physical link between the simulator and the flight controller — converting software attitude commands into real 6-DOF motion so the PX4 IMU actually moves.

I'll show you what this physical stimulus reveals, with real data. But first, let's see the system in action.

## Slide 4 — System in action

[Play demo video]

What you're seeing is our baseline experiment — a trim transition scenario. X-Plane commands a pitch change from 0 to +5 degrees, the Stewart platform tilts accordingly, and the PX4 mounted on top senses this motion in real time.

We'll see the tracking data from exactly this type of run in the results. But first, let me walk through the system architecture.

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

X-Plane computes the dynamics and publishes vehicle state over UDP. In a conventional setup the loop closes inside the simulator, but here PX4 attitude returns through the physical path.

## Slide 7 — Block 2: Python host / bridge

[Architecture figure with Block 2 highlighted]

The host is the integration hub. It receives X-Plane state and PX4 attitude simultaneously, converts simulator state to platform commands, applies common timestamps for logging, and runs the PID controller that closes the loop back to X-Plane.

## Slide 8 — Block 3: Stewart platform actuation

[Architecture figure with Block 3 highlighted]

Takes the host's pose command and converts it to real physical motion. We used 50 Hz hobby-grade RC servos to minimize cost; serial command-ACK round trip is about 22 ms. After this block, what PX4 experiences is actual inertial motion.

## Slide 9 — Block 4: PX4 sensing and return

[Architecture figure with Block 4 highlighted]

PX4's onboard IMU senses the platform's real motion and its estimator computes attitude. That estimate returns to the host, where the PID controller generates control-surface commands for X-Plane, completing the loop.

## Slide 10 — What we tested and how we measured it

Now, what did we actually test?

We set the platform's neutral position at z = +20 mm — that maximizes the available rotation range.

The baseline scenario is a simple trim transition: pitch goes from 0 to +5 degrees, holds there, then comes back. Every run launches from the same X-Plane starting condition — a Honolulu PHNL 10-nautical-mile approach.

Each run monitors the saturation flag throughout, and we ran 9 repetitions per condition for statistical confidence. We need to isolate system-inherent latency, bias, and mounting effects first, so we start with a clean baseline under controlled conditions.

## Slide 11 — Result 1: Closed-loop tracking is achieved

Now let's look at the results.

[Show Fig. 2: pitch response overlay]

The orange line is PX4 pitch, the blue dashed line is X-Plane commanded pitch. You can see PX4 tracks the trim transition. But notice the constant gap of about +1.8 degrees between the two — that is the mounting bias. The solid blue line is X-Plane pitch shifted by that bias; once corrected, the tracking becomes much tighter.

The key point: closed-loop tracking works, and the dominant error is physically explainable.

## Slide 12 — Result 2: Mounting quality explains the dominant error

So if there's a bias, where does it come from? We validate the mounting-quality hypothesis with two pieces of evidence.

[Show Fig. 3: pitch error + Fig. 4: mounting bias evidence]

First evidence: when we change mounting quality, bias behavior changes. With loose mounting (adhesive tape only), bias grows over time as the FC physically shifts under repeated inertial loading. With firm mounting (structural adhesive plus mechanical tiedowns), that drift disappears.

Second evidence: in the firm-mount condition, the remaining bias stays nearly constant through the run. That means the residual error is not dynamic instability; it's a static alignment offset from mounting.

Taken together, the dominant pitch error comes from mounting quality, not the control system. Once mounting is fixed, error converges to a static bias that post-processing can automatically estimate and compensate.

## Slide 13 — Result 3: Latency characterization and system performance

We showed tracking works — but how fast can this system actually respond? This slide answers that.

[Show Fig. 5(a): latency time series + Fig. 5(b): latency distribution]

We decomposed latency into four segments. X-Plane sample age is how stale the simulator data is when the host uses it. PX4 sample age is how stale PX4's attitude estimate is when the host reads it. Serial RTT is the command-to-ACK round trip between the host and the Stewart platform. And end-to-end latency is the total time from X-Plane data arrival to completed platform actuation.

On the left time series, these are shown as blue, orange, green, and red respectively. Solid lines are binned means, shading is the min-max range. The red end-to-end line hovers around 40 to 60 ms.

On the right, the box plot shows the distribution across all five baseline runs. End-to-end median is about 48 ms, with X-Plane age around 27 ms, PX4 age around 19 ms, and serial RTT around 22 ms overlapping in the pipeline.

This latency chain sets the practical limit on dynamic fidelity. The jitter spread is significant enough that we cannot pin down an effective bandwidth from these data alone. That will come from a dedicated sine-sweep measurement.

## Slide 14 — Takeaways

So to wrap up — three things I want you to take away.

First, integration. We built a physically closed-loop HILS that connects X-Plane, a Stewart platform, and PX4 into one working system.

Second, fidelity. Bias-corrected pitch tracking RMSE is about 0.46 degrees, with end-to-end latency around 51 ms.

Third, by characterizing the system — its latency, bias, and workspace limits — we established a quantitative baseline for future expansion to disturbance testing and other vehicle types.

That said, this system does not replace conventional software HILS or real flight testing. The platform has a finite workspace, it cannot reproduce aerodynamic loads, and effective bandwidth still needs dedicated measurement. We see this as a complementary validation layer between software HILS and flight test.

For next steps: in the near term, time-aligned error analysis, sine-sweep-based bandwidth measurement, repeatability characterization, and stress tests near the saturation boundary. Longer term, we want to extend to other vehicle types — multirotor, spacecraft — to demonstrate the architecture's generality.

## Slide 15 — End

Thank you. I'm happy to take any questions.

## Slide 16 — Backup: z=+20 mm proof figure

Navigate here only if the z=+20 mm question comes up. Otherwise stay on Slide 15 and navigate to main-body slides as indicated in the Q&A prepared answers below.

---

## Optional live delivery cues

- Slides 6–9 (block details) are condensed to ~15–20 seconds each. Visuals carry the detail; spoken script gives only the key role per block.
- Spend most technical depth on Slides 11 (tracking), 12 (bias), and 13 (latency).
- Slide 4 (demo video): narrate the trim-transition scenario as it plays — 15–20 seconds, then bridge to "we'll see this data in the results."

---



## Q&A prepared answers (English)

Each answer includes a presentation navigation cue in brackets.

**"Why z=+20 mm as the neutral position?"**
[Navigate to slide 16 — proof figure available]
The available rotation range depends on z height. At z=+20 mm, the available range in that neutral position is near its maximum; the firmware's worst-direction rotation limit is about 20° there, versus about 10° around z=0 mm. This maximizes the usable attitude envelope.

**"Why not just conventional HILS?"**
[Navigate to slide 3 — key differentiator slide]
We add real inertial excitation that sensor synthesis cannot provide, while honestly acknowledging workspace and aerodynamic reproduction limits.

**"How do you handle saturation?"**
[Navigate to slide 10 — experiment setup slide]
We compare the baseline commands against the available rotation range at z=+20 mm and monitor the Stewart saturation flag throughout. In the refreshed baseline metrics, observed saturation is 0-1% depending on the run, so the baseline mostly represents nominal-region behavior. Dedicated stress tests near the saturation boundary are still planned.

**"What about yaw?"**
[Stay on slide 15 — verbal answer]
Yaw is in the loop, but the trim-transition scenario primarily excites pitch. Yaw tracking and bias alignment are handled in the code but not the focus of this presentation.

**"What's the effective bandwidth of this system?"**
[Navigate to slide 13 — show latency data while answering]
Today I can support only the measured end-to-end latency of about 48 ms. The effective attitude bandwidth will be reported from a dedicated sine-sweep measurement. Until that test is complete, I describe the system only as suited to low-frequency maneuvers such as trim transitions.

**"The platform is open-loop — how do you trust pose accuracy?"**
[Stay on slide 15 — verbal answer]
A design goal was to keep the Stewart platform as affordable as possible, which led to using inexpensive RC aircraft-grade hobby servos without encoder feedback — hence the open-loop actuation. Despite this, we measured the platform's static pose accuracy with a digital inclinometer: approximately 0.1° in roll and 0.5° in pitch. So we have a practical bound on platform-level error even without continuous feedback. The PX4 IMU remains the only closed-loop physical measurement; we measure the output (attitude) but not the intermediate stage (platform pose). Adding encoders is a possible future improvement.

**"What is the mounting bias physically? Can you eliminate it?"**
[Navigate to slide 12 — show Fig 3/4 while answering]
The ~1.8° bias comes from the physical alignment of the FC on the platform top plate. It is consistent within a run (std ~0.2°) and can be calibrated out in post-processing, which we do. The physical mount could also be shimmed or re-aligned, but for HILS validation purposes, software correction is sufficient as long as the bias is stable.

**"How repeatable are the results across runs?"**
[Navigate to slide 12 — show Fig 4 while answering]
We have 5 baseline runs (#5–#9). Within each run, bias std is ~0.2°. Across runs, the bias estimate varies by up to 0.8°, likely due to thermal drift and minor re-seating between runs. Full repeatability characterization is listed as a next step.

**"Can this extend to quadrotors or other vehicles?"**
[Navigate to slide 14 — show future plans while answering]
In principle, yes — the architecture is not tied to fixed-wing only. But this talk demonstrates fixed-wing as the first case, and cross-vehicle validation should be treated as future work until we actually run those vehicles on the platform.

**"Can you show how the FC is mounted on the platform?"**
[Navigate to slide 12 — redirect to bias data]
The mounting comparison itself is the scientific evidence — when we change mounting condition, bias behavior changes predictably. Figures 3 and 4 show this: loose mount gives progressive drift, firm mount gives a stable static offset. The physical mount is intentionally simple and low-cost, consistent with the system's overall design philosophy. The quantitative bias data is more informative than a photograph of the mount.
