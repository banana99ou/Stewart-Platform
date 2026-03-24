# Presentation content and framing feedback
_Exported on 3/22/2026 at 17:49:15 GMT+9 from Cursor (2.6.19)_

---

**User**

I'm going to send PI this message with @presentation/압축_발표_대본.pdf @presentation/KSAS_Spring2026_Presentation.pdf. does this message have enough context? if not please rewrite.
```
안녕하세요 교수님.
이번 발표 자료를 1차로 정리해보았습니다.
현재 구성에서 한 가지는 연구 framing, 다른 한 가지는 Stewart platform 관련 내용을 어디까지 포함할지에 대해 의견을 여쭙고 싶습니다.

첫째, 전체 framing을 broader한 physical HILS로 가져갈지, 아니면 IMU-in-the-loop 기반 estimator validation 중심으로 좁히는 것이 맞을지 의견 부탁드립니다.

둘째, Stewart platform이 custom 제작물이다 보니 실험 타당성과 직접 연결되는 내용들, 예를 들면

working area와 z 위치 관계

중립 z 선정 근거

command saturator 및 saturation monitoring
같은 부분은 어느 정도까지 설명하는 것이 적절할지 판단 부탁드립니다.

제 생각에는 개발 과정 자체를 길게 설명하기보다는, 실험의 유효 범위와 제약을 정당화하는 수준까지만 넣는 것이 맞을 것 같다고 생각하고 있습니다.
현재 결과 수준에서 어디까지 포함하는 것이 가장 적절한지 조언 부탁드립니다.
```

---

**Cursor**

요청 자체는 합리적입니다(PI가 “어느 방향으로 framing할지”와 “Stewart platform을 어디까지 설명할지”를 명확히 의사결정하도록 유도하고 있어요). 다만 현재 문구는 **(1) 첨부 파일이 어떤 “버전/목적”인지**, **(2) 교수님께서 주시면 좋은 피드백의 범위(예: 구조/깊이/우선순위)**, **(3) 반영을 위한 시간 제약(가능하시면 언제까지)** 같은 컨텍스트가 조금 부족해서, PI가 “그래서 어디를 특히 봐주면 되는지”를 빠르게 파악하기가 어려울 수 있습니다.

아래처럼 첨부물의 성격(1차 초안)과 요청의 초점을 한 줄로 더 보강하면 컨텍스트가 더 충분해질 것 같습니다.

```text
안녕하세요 교수님.
첨부드린 `@presentation/압축_발표_대본.pdf`와 `@presentation/KSAS_Spring2026_Presentation.pdf`는 이번 발표의 1차 초안(구성/서술 방향 정리본)입니다.
초안을 기반으로 아래 두 가지를 확인하고자 의견을 부탁드립니다.

1) 연구 framing 방향에 대해
전체 framing을 broader한 physical HILS로 가져갈지, 아니면 IMU-in-the-loop 기반 estimator validation 중심으로 좁히는 것이 더 적절할지 교수님 의견을 여쭙고 싶습니다.

2) Stewart platform 설명 범위에 대해
working area와 z 위치 관계, 중립 z 선정 근거, command saturator 및 saturation monitoring처럼 실험 타당성과 직접 연결되는 내용들은 어느 정도까지 포함하는 것이 적절할지 판단 부탁드립니다.
제 생각에는 개발 과정 자체를 길게 설명하기보다는, 실험의 유효 범위와 제약을 정당화하는 수준에서 멈추는 것이 맞는 방향이라고 보고 있습니다. 교수님께서도 이 방향이 타당하실지 조언 부탁드립니다.

가능하시면 초안에서 특히 유지/추가/삭제했으면 하는 포인트가 있으면 표시해주시면 반영하겠습니다.
감사합니다.
```

원하시면, 교수님께 드리는 톤(더 포멀/더 간결)과 “검토 희망 시점(예: 다음 미팅 전)”을 알려주시면 그에 맞춰 한 버전 더 다듬어드릴게요.

---

**User**

here's prof's feedback.

1. save it to doc.
2. analyze for key point and simple editing
3. analyze for overall accessment of PI about this script.

```
1. 둘 다 좋아 보이는데 정현용학생이 학회에서 참가자분들과 논의하기 편리한 것으로 하시면 됩니다.
2. 학회 발표시에는 세부적인 디테일은 그냥 표 정도로 넣고 참고용으로 제시만 하시고 시시콜콜 설명하지 않는 편이 낫겠습니다. 프로젝트의 스토리텔링에 집중해서 청중들이 흥미로워할 수 있는 부분을 강조해 주세요.
3. 초안에서 사진이 아직 없어서 아쉬운데 작동 동영상도 있으면 좋겠습니다. 단순히 작동하는거 봤지? 가 아니라 특정 시나리오에 대한 작동 영상을 보여주면서 움직님을 설명할 수 있는 것으로요. 
4. 폰트가 너무 얇아서 잘 안보일것같습니다. 발표 환경에 강건하게 잘 보이는 폰트로 하시고 큼직큼직하게 해주세요. 굳이 안 읽어도 전체 흐름 이해에 문제없는 내용만 작은 글씨로 하세요. 
5. 5페이지 블록 요소별로 대표 사진 + 다른 요소와 주고받는 정보에 대한 상세설명 슬라이드를 만들어 주세요. 깔끔하게 하시긴 했는데, 처음 보는 입장에서 한 슬라이드에 들어가는 정보가 너무 많으므로 분산시키면 좋겠습니다.
6. 각 그래프들 배치는 널찍널찍하게 잘 하셨습니다만, 실선 굵기를 좀 더 굵게 해서 잘 보이게 하시고 겹치는 부분이 많은 그래프는 마커나 점선 등으로 확실히 구분되게 해주세요.
7. 표지에 국문 제목도 적어주세요.
```

---

**Cursor**

Let me first read the relevant files for context.

Let me check the existing feedback doc for format reference.

Now I have good context. Let me save the feedback and then provide my analysis.

Saved. Now here's the analysis:

---

## 2. Key Points & Simple Editing Actions

| # | Key Point | Concrete Action | Complexity |
|---|-----------|----------------|------------|
| 1 | **Framing: your call** — PI is fine with either direction. Pick whichever makes post-talk Q&A easier for you. | Decision only, no slide change needed yet. | None |
| 2 | **Kill verbose detail, use tables** — Technical specs (working area, z position, saturator, etc.) should be in a reference table, not narrated. Focus oral delivery on *storytelling*. | Move specs from narrated content into a summary table slide. Strip those details from the script. | Medium |
| 3 | **Add photos + scenario-driven demo video** — Not just "look it moves," but tied to a specific scenario (e.g., the trim transition) with narration over motion. | Record a video of the trim-transition run with X-Plane side-by-side; embed or link in slide 4. Add hardware photos where missing. | High (production) |
| 4 | **Font: thicker, bigger** — Thin font won't survive projector/lighting. Only non-essential info can be small. | In `generate_presentation.py`, switch to a bolder font weight and increase base font sizes. | Low-medium |
| 5 | **Slide 5 (system block diagram) is overloaded** — Break into per-block detail slides: each block gets its own photo + data-flow annotation. | Split slide 5 into 3-4 sub-slides (X-Plane, Host/Simulink, Stewart Platform, PX4), each with a representative photo and interface details. | Medium |
| 6 | **Graph lines too thin; overlapping traces need differentiation** — Thicker lines, add markers or dashed styles. | In `Fig_Gen.py`, increase `linewidth`, add markers/linestyles where traces overlap (especially Fig2 pitch overlay and Fig5a latency). | Low |
| 7 | **Korean title on cover slide** — Add 국문 제목. | Add the Korean title (from the paper: "스튜어트 플랫폼을 이용한 항공우주 비행체 자세제어 및 HILS 플랫폼 개발") to the title slide. | Trivial |

---

## 3. Overall Assessment of PI

**Tone: Positive with targeted corrections.** The PI is not questioning the research direction, the data, or the methodology. There is no pushback on the results, no request to redo experiments, and no concern about validity. That's a strong signal.

What the feedback *is* about:

- **Presentation craft, not research substance.** Every single point (1-7) is about how you *show* the work, not about what the work *is*. This means the PI considers the technical content settled enough for conference presentation.

- **PI trusts your judgment on framing** (point 1). Delegating the framing choice to you suggests confidence that you understand the material well enough to pick the right angle for the audience.

- **The main concern is audience experience.** Points 2, 4, 5, 6 all converge on the same meta-feedback: *the slides are currently designed for someone who already knows the project, not for a first-time viewer.* The PI wants you to shift perspective from "technically complete" to "audience-accessible."

- **The video request (point 3) is the highest-value item.** The PI specifically emphasized *scenario-driven* demonstration, not just a generic hardware clip. This is likely something the PI thinks will make the talk memorable and differentiate it from a pure-data presentation.

**Bottom line:** The PI is satisfied with the research and the draft structure. The revisions are about *polish and accessibility* — making the talk land well with a conference audience who sees it for the first time. No red flags, no structural rework needed.

---

**User**

please update @ToDo.md 

---

**Cursor**

Let me read the current ToDo.md and the presentation script to understand what needs updating.

Now I'll update the ToDo.md with the new PI feedback items.

Let me verify the result looks correct.

Done. Added **P0.5 — Presentation revision per PI feedback (2026-03-20)** (lines 65–86) with 7 actionable items mapped directly from the PI's 7 feedback points. Each item has sub-tasks where applicable, and the section links back to `doc/Prof_feedback_2026-03-20_presentation.md` for the full original text.

