## Ranked reference candidates (TL;DR + why to cite)

> Note: verify exact bibliographic metadata (year/venue/pages/report number) before final submission.

### 1) Merlet — *Parallel Robots* (book)
- **TL;DR**: Standard reference for parallel manipulators: kinematics, singularities, workspace, calibration, and practical implementation issues.
- **Why cite**: One citation covers most “parallel robot / Stewart platform theory” questions without over-explaining.

### 2) Stewart (1965) — “A Platform with Six Degrees of Freedom”
- **TL;DR**: Introduces the 6‑DOF platform concept that later became the Stewart platform.
- **Why cite**: Establishes mechanism lineage and legitimacy with a universally recognized reference.

### 3) Dasgupta & Mruthyunjaya — “The Stewart Platform Manipulator: A Review”
- **TL;DR**: Survey summarizing Stewart platform modeling, IK approaches, applications, and common constraints.
- **Why cite**: Lets you cite a single review for “prior IK / platform research exists” instead of many scattered papers.

### 4) Telban & Cardullo (NASA TR, 2005) — motion cueing algorithm development
- **TL;DR**: Practical, engineering-focused motion cueing discussion with evaluation considerations (fidelity/latency tradeoffs).
- **Why cite**: Supports motivation (“motion cueing matters”) and helps justify evaluation dimensions without claiming you implemented a specific washout.

### 5) Sivan / Ish‑Shalom / Huang (1982) — optimal-control motion cueing
- **TL;DR**: Formulates motion cueing as an optimization problem under platform constraints.
- **Why cite**: Anchors your “limited workspace → saturation → graceful degradation” story with a strong theory reference.

### 6) Reid & Nahon — motion-base drive / washout algorithms (Part I/II)
- **TL;DR**: Classic systematic treatment of washout/motion-base drive concepts and constraint handling.
- **Why cite**: Strong background if you discuss motion platform behavior, constraints, and what “good motion reproduction” means.

### 7) Isermann / Schaffnit / Sinsel (1999) — Hardware-in-the-loop simulation methodology
- **TL;DR**: Canonical HIL methodology paper: why HIL is used and what to validate (timing, interfaces, robustness).
- **Why cite**: Gives credibility to HIL evaluation/verification framing (even if the example domain is not aerospace).

### 8) Meier / Honegger / Pollefeys (2015) — PX4 platform paper
- **TL;DR**: Describes PX4 as an open, modular research autopilot platform and motivates its use in research/validation.
- **Why cite**: Justifies PX4/Pixhawk as a legitimate research-grade FC choice.

### 9) MAVLink Protocol Specification (official spec/docs)
- **TL;DR**: Defines MAVLink messages and transport used to exchange attitude/state/actuator info.
- **Why cite**: Authoritative reference for your PX4↔host communications layer.

### 10) X‑Plane data/UDP documentation (Laminar Research)
- **TL;DR**: Documents X‑Plane state output and control input mechanisms (UDP/data output/SDK).
- **Why cite**: Authoritative reference for your simulator interface.

## If you only keep 5 references (≤5 constraint)
Recommended top‑5 for a 2‑page abstract:
1) Merlet (*Parallel Robots*)  
2) Stewart (mechanism origin)  
3) Dasgupta & Mruthyunjaya (review)  
4) Telban & Cardullo (motion cueing + evaluation context)  
5) Isermann et al. (HIL methodology) **or** Meier et al. (PX4 platform credibility)


