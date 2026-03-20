"""Generate KSAS Spring 2026 conference PowerPoint from the blue Canva template."""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "fig")
TEMPLATE = os.path.join(BASE, "template.pptx")

# Use a Korean-capable font for all generated text.
FONT_NAME = "Nanum Square"

# ── Colors (from template: grey bg, white panel, blue+black text) ───

BG_GREY = RGBColor(0xDC, 0xDC, 0xDC)
BLUE = RGBColor(0x00, 0x4A, 0xAD)
BLUE_PALE = RGBColor(0xD0, 0xE0, 0xF5)
HEADER_PALE = RGBColor(0xEA, 0xF3, 0xFF)  # template header block color
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK = RGBColor(0x2D, 0x2D, 0x2D)
GREY = RGBColor(0x66, 0x66, 0x66)
KO_GREY = RGBColor(0x99, 0x99, 0x99)
LIGHT = RGBColor(0xF0, 0xF0, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC0, 0x39, 0x2B)
RED_LIGHT = RGBColor(0xFA, 0xE5, 0xE3)

# ── Dimensions (template 20" × 11.25") ─────────────────────────────

W = Inches(20)
H = Inches(11.25)
# White content panel matching template's Group 2 (full-height variant)
PL, PT_, PW, PH = Inches(0.88), Inches(0.83), Inches(18.23), Inches(9.44)
# Header band occupies top 2.11" of the white panel
HEADER_H = Inches(2.11)
# Content zone starts right below the header
CONTENT_TOP = Inches(0.83 + 2.11 + 0.3)  # ~3.24"
CL = Inches(1.6)
CW = Inches(16.8)
TOTAL = 13


# ── Helpers ─────────────────────────────────────────────────────────

def _rr(sl, l, t, w, h, fill, line=None):
    s = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def _rounded_rect(sl, l, t, w, h, fill, line=None):
    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def _tb(sl, l, t, w, h):
    tb = sl.shapes.add_textbox(l, t, w, h)
    # Set default font on the first paragraph so direct `p = ...paragraphs[0]`
    # formatting later inherits the same font.
    try:
        p0 = tb.text_frame.paragraphs[0]
        p0.font.name = FONT_NAME
    except Exception:
        pass
    return tb


def _set(shape, text, sz=22, color=DARK, bold=False, align=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(sz)
    p.font.name = FONT_NAME
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = align
    return tf


def _p(tf, text, sz=22, color=DARK, bold=False, sp=Pt(8), align=PP_ALIGN.LEFT):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(sz)
    p.font.name = FONT_NAME
    p.font.color.rgb = color
    p.font.bold = bold
    p.space_before = sp
    p.alignment = align
    return p


def _ko(tf, text, sz=16, sp=Pt(2), align=PP_ALIGN.LEFT):
    return _p(tf, text, sz, KO_GREY, False, sp, align)


def _bullet(tf, en, kr=None, sz=20, sp=Pt(10)):
    _p(tf, f"\u2022  {en}", sz, DARK, sp=sp)
    if kr:
        _ko(tf, f"    {kr}", sz - 4, Pt(2))


def _img(sl, path, l, t, width=None, height=None):
    if os.path.exists(path):
        return sl.shapes.add_picture(path, l, t, width, height)
    print("path not found", path)
    box = _rr(sl, l, t, width or Inches(6), height or Inches(4), LIGHT, GREY)
    _set(box, f"[{os.path.basename(path)}]", 18, GREY, align=PP_ALIGN.CENTER)
    return box


def _emu_to_in(v):
    return v / 914400.0


def _sn(sl, n):
    tb = _tb(sl, Inches(18.2), Inches(10.55), Inches(1.2), Inches(0.5))
    p = tb.text_frame.paragraphs[0]
    p.text = f"{n}/{TOTAL}"
    p.font.size = Pt(14)
    p.font.color.rgb = GREY
    p.alignment = PP_ALIGN.RIGHT


def _circle(sl, l, t, d, fill, text, text_color=WHITE, text_sz=20):
    c = sl.shapes.add_shape(MSO_SHAPE.OVAL, l, t, d, d)
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    _set(c, text, text_sz, text_color, True, PP_ALIGN.CENTER)
    c.text_frame.margin_top = Pt(2)
    c.text_frame.margin_left = Pt(0)
    c.text_frame.margin_right = Pt(0)
    return c


def make_slide(prs, layout):
    """Grey bg + white content panel, matching the template."""
    sl = prs.slides.add_slide(layout)
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = BG_GREY
    _rr(sl, PL, PT_, PW, PH, WHITE)
    return sl


F2F2 = RGBColor(0xF2, 0xF2, 0xF2)


def slide_title(sl, text, y=PT_):
    """Template #EAF3FF header band (18.23" x 2.11") with centered title at y=1.82"."""
    header = _rr(sl, PL, y, PW, HEADER_H, HEADER_PALE)
    header.shadow.inherit = False

    tb = _tb(sl, Inches(5.34), Inches(1.5), Inches(9.31), Inches(0.73))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(40)
    p.font.color.rgb = BLACK
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER


def _callout(sl, l, t, w, h, text, sz=18):
    """Template-style #EAF3FF rounded callout box with bold text."""
    box = _rounded_rect(sl, l, t, w, h, HEADER_PALE)
    box.shadow.inherit = False
    tb = _tb(sl, l + Inches(0.4), t + Inches(0.4), w - Inches(0.8), h - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(sz)
    p.font.color.rgb = BLACK
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    return tf


def _card(sl, l, t, w, h, fill=F2F2):
    """Template-style #F2F2F2 rounded card background."""
    c = _rounded_rect(sl, l, t, w, h, fill)
    c.shadow.inherit = False
    return c


# ── Load template, clear slides ─────────────────────────────────────

prs = Presentation(TEMPLATE)
while len(prs.slides._sldIdLst) > 0:
    rId = prs.slides._sldIdLst[0].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[0]

blank = prs.slide_layouts[6]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 1 — Title
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)

tb = _tb(sl, Inches(2), Inches(1.2), Inches(16), Inches(1.0))
p = tb.text_frame.paragraphs[0]
p.text = "PHYSICALLY CLOSED-LOOP HILS"
p.font.size = Pt(24)
p.font.color.rgb = BLUE
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

tb = _tb(sl, Inches(2), Inches(2.0), Inches(16), Inches(3.5))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Physically Closed-Loop HILS for\nFlight Attitude-Control Validation"
p.font.size = Pt(56)
p.font.color.rgb = BLACK
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.line_spacing = Pt(68)

_p(tf, "Using a Stewart Platform with X-Plane and PX4", 22, GREY, sp=Pt(24), align=PP_ALIGN.CENTER)
_ko(tf, "스튜어트 플랫폼 · X-Plane · PX4 연동 기반 비행체 자세제어 검증",
    18, Pt(6), PP_ALIGN.CENTER)

tb = _tb(sl, Inches(2), Inches(7.0), Inches(16), Inches(1.0))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Spring 2026 KSAS Conference"
p.font.size = Pt(22)
p.font.color.rgb = BLACK
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
_p(tf, "\uc815\ud604\uc6a9 \u2014 \uad6d\ubbfc\ub300\ud559\uad50 \ubbf8\ub798\ubaa8\ube4c\ub9ac\ud2f0\uc81c\uc5b4\uc5f0\uad6c\uc2e4", 18, GREY, sp=Pt(6), align=PP_ALIGN.CENTER)

_sn(sl, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 2 — Motivation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "Why Add Physical Motion to HILS?")

ko_tb = _tb(sl, PL, Inches(2.65), PW, Inches(0.5))

col_w = Inches(7.5)
col_top = Inches(3.4)
lx = Inches(1.8)
rx = Inches(10.7)

# Conventional HILS column
ltb = _tb(sl, lx, col_top, col_w, Inches(5.5))
ltf = ltb.text_frame
ltf.word_wrap = True
p = ltf.paragraphs[0]
p.text = "Conventional HILS"
p.font.size = Pt(28)
p.font.color.rgb = BLUE
p.font.bold = True
for en, kr in [
    ("Simulator computes dynamics", "시뮬레이터가 동역학 계산"),
    ("Controller receives synthetic sensor signals", "합성 센서 신호 수신"),
    ("Attitude evaluated via logs / screens", "로그/화면으로만 자세 확인"),
    ("IMU does not experience real motion", "IMU가 실제 관성 운동 미경험"),
]:
    _bullet(ltf, en, kr, 20, Pt(14))

# This Work column
rtb = _tb(sl, rx, col_top, col_w, Inches(5.5))
rtf = rtb.text_frame
rtf.word_wrap = True
p = rtf.paragraphs[0]
p.text = "This Work"
p.font.size = Pt(28)
p.font.color.rgb = BLUE
p.font.bold = True
for en, kr in [
    ("Simulator drives real Stewart platform motion", "실제 플랫폼 모션 구동"),
    ("PX4 IMU senses real inertial excitation", "PX4 IMU 실제 관성 자극 감지"),
    ("Measured attitude fed back into loop", "측정 자세 → 제어 루프 피드백"),
    ("Timing and fidelity measured", "타이밍·충실도 정량 측정"),
    ("Vehicle-agnostic: quad, satellite, missile...", "기체 비종속: 쿼드콥터, 위성, 유도탄 등"),
]:
    _bullet(rtf, en, kr, 20, Pt(14))

# divider line between columns
div = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.85), Inches(3.6), Pt(2), Inches(4.8))
div.fill.solid()
div.fill.fore_color.rgb = BLUE
div.line.fill.background()

# bottom callout
_callout(sl, Inches(2.67), Inches(8.8), Inches(14.66), Inches(1.2),"물리적 피드백 포함 HILS 구조의 안정성, 반복성, 정량적 설명 가능성 제시", 20)

_sn(sl, 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 3 — What Is New
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "What Is New Here?")

ctb = _tb(sl, Inches(2), Inches(3.2), Inches(16), Inches(1.2))
tf = ctb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = ("시뮬레이션 센서 값뿐만 아니라 실제 IMU 모션 기반 자세 추정치를 폐루프 피드백 입력으로 활용")
p.font.size = Pt(22)
p.font.color.rgb = DARK
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

bw = Inches(5.0)
bh = Inches(3.2)
btop = Inches(4.8)
gap = Inches(0.5)

total_w = 3 * bw + 2 * gap
left = (prs.slide_width - total_w) // 2
bxs = [left + i * (bw + gap) for i in range(3)]

labels = [
    ("01", "Realized Motion", "모션 구현",
     "Simulator attitude is converted\ninto Stewart-platform motion.",
     "시뮬레이터 자세 → 플랫폼 모션"),
    ("02", "Physical Sensing", "물리적 센싱",
     "PX4 IMU and estimator respond\nto real inertial excitation.",
     "PX4 IMU가 실제 관성 자극에 반응"),
    ("03", "Closed-Loop Return", "폐루프 귀환",
     "Measured attitude is sent back\ninto the host control loop.",
     "측정된 자세 → 호스트 제어 루프"),
]

for i, (num, t_en, t_kr, d_en, d_kr) in enumerate(labels):
    x = bxs[i]

    ntb = _tb(sl, x, btop, bw, Inches(0.5))
    p = ntb.text_frame.paragraphs[0]
    p.text = num
    p.font.size = Pt(30)
    p.font.color.rgb = BLUE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    ttb = _tb(sl, x, btop + Inches(0.6), bw, Inches(0.5))
    p = ttb.text_frame.paragraphs[0]
    p.text = t_en
    p.font.size = Pt(22)
    p.font.color.rgb = DARK
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    dtb = _tb(sl, x, btop + Inches(1.1), bw, Inches(1.8))
    dtf = dtb.text_frame
    dtf.word_wrap = True
    p = dtf.paragraphs[0]
    p.text = d_en
    p.font.size = Pt(18)
    p.font.color.rgb = DARK
    p.alignment = PP_ALIGN.CENTER
    _ko(dtf, d_kr, 14, Pt(8), PP_ALIGN.CENTER)

ctf = _callout(
    sl, Inches(2.67), Inches(8.5), Inches(14.66), Inches(1.6),
    "기체 비종속 구조: 시뮬레이터 모델과 제어기 튜닝만 변경하면 멀티로터, 위성 등 다양한 기체를 동일 환경에서 검증 가능", 18
)
_p(ctf, "보완적 검증 수단 - 전 영역 공력 효과 재현이 아닌, 폐루프 성능 중심 설계",
   16, GREY, sp=Pt(6), align=PP_ALIGN.CENTER)

_sn(sl, 3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 4 — System in Action
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "System in Action")

pw = Inches(7.8)
ph = Inches(5.0)
photo = _rr(sl, Inches(1.5), Inches(3.4), pw, ph, LIGHT, GREY)
photo.text_frame.word_wrap = True
photo.text_frame.margin_top = Inches(2.0)
photo.text_frame.margin_left = Pt(20)
_set(photo, "[Insert hardware photo]", 22, GREY, align=PP_ALIGN.CENTER)
_ko(photo.text_frame, "스튜어트 플랫폼 + PX4 탑재 사진",
    16, Pt(8), PP_ALIGN.CENTER)

video = _rr(sl, Inches(10.5), Inches(3.4), pw, ph, LIGHT, GREY)
video.text_frame.word_wrap = True
video.text_frame.margin_top = Inches(2.0)
video.text_frame.margin_left = Pt(20)
_set(video, "[Insert demo video]", 22, GREY, align=PP_ALIGN.CENTER)
_ko(video.text_frame, "Trim transition 시나리오 구동 영상",
    16, Pt(8), PP_ALIGN.CENTER)

_callout(sl, Inches(2.67), Inches(8.8), Inches(14.66), Inches(1.2),
         "X-Plane sim attitude \u2192 Stewart platform motion \u2192 "
         "PX4 IMU sensing \u2192 closed-loop", 20)

_sn(sl, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 5 — System Architecture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "How the Physical Closed Loop Works")

fig1 = os.path.join(FIG, "Fig1_Architecture.png")
fig_left = Inches(2.7)
fig_top = Inches(3.4)
pic = _img(sl, fig1, fig_left, fig_top, height=Cm(13.0))

IMG_W_PX = 1646
img_w_in = _emu_to_in(pic.width) if hasattr(pic, "width") and pic.width else 10.5
px_per_inch = IMG_W_PX / img_w_in
cd = Inches(0.55)
for num, px_x, px_y in [(1, 190, 180), (2, 190, 555), (3, 730, 850), (4, 1420, 555)]:
    cx = fig_left + Inches(px_x / px_per_inch) - cd / 2
    cy = fig_top + Inches(px_y / px_per_inch) - cd / 2
    c = _circle(sl, cx, cy, cd, RED, str(num), WHITE, 20)
    c.line.color.rgb = WHITE
    c.line.width = Pt(2)

callout_x = Inches(12.2)
callout_top = Inches(3.4)
steps = [
    ("1", "X-Plane publishes vehicle state\nto the Python host over UDP."),
    ("2", "Host converts state to pose\ncommands for the Stewart platform."),
    ("3", "PX4 physically senses platform\nmotion through its onboard IMU."),
    ("4", "PX4 attitude returns to host;\nhost injects control into X-Plane."),
]
for i, (num, en) in enumerate(steps):
    y = callout_top + Inches(i * 1.3)

    ntb = _tb(sl, callout_x, y, Inches(0.6), Inches(0.5))
    p = ntb.text_frame.paragraphs[0]
    p.text = num
    p.font.size = Pt(30)
    p.font.color.rgb = BLUE
    p.font.bold = True

    tb = _tb(sl, callout_x + Inches(0.7), y+Cm(0.1), Inches(6.0), Inches(1.3))
    ttf = tb.text_frame
    ttf.word_wrap = True
    p = ttf.paragraphs[0]
    p.text = en
    p.font.size = Pt(24)
    p.font.color.rgb = DARK
    p.line_spacing = Pt(24)

ftb = _tb(sl, Inches(2), Inches(9.3), Inches(16), Inches(0.6))
p = ftb.text_frame.paragraphs[0]
p.text = "SW·기구부·센싱·제어 I/O를 하나로 잇는 물리적 폐루프"
p.font.size = Pt(20)
p.font.color.rgb = BLUE
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

_sn(sl, 5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 6 — Experiment Design
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "What We Tested and How We Measured It")

col_w = Inches(7.5)
zone_top = Inches(3.4)
lx = Inches(1.8)
rx = Inches(10.7)

ltb = _tb(sl, lx, zone_top, col_w, Inches(6.5))
ltf = ltb.text_frame
ltf.word_wrap = True
p = ltf.paragraphs[0]
p.text = "Baseline Scenario"
p.font.size = Pt(26)
p.font.color.rgb = BLUE
p.font.bold = True
_p(ltf, "Pitch trim transition:", 22, DARK, True, Pt(18))
_p(ltf, "0\u00b0 \u2192 +5\u00b0 \u2192 hold \u2192 0\u00b0", 26, BLUE, True, Pt(6))
_p(ltf, "", 8, sp=Pt(8))
_bullet(ltf, "항상 동일 Honolulu (PHNL) 10nm 어프로치로 시작", "", 18, Pt(8))
_p(ltf, "", 8, sp=Inches(0.09))
_bullet(ltf, "스크립트로 적용된 유예 기간 동안 시스템 안정화 후 데이터 수집", "", 18, Pt(8))
_p(ltf, "", 8, sp=Inches(0.09))
_bullet(ltf, "Stewart 플랫폼의 포화 플래그(SAT) 감시; 작업 공간 내 유지", "", 18, Pt(8))
_p(ltf, "", 8, sp=Inches(0.09))
_bullet(ltf, "조건별 9회 반복 실행으로 통계적 유의미성 확보", "", 18, Pt(8))
_p(ltf, "", 8, sp=Inches(0.09))
_bullet(ltf, "z=+20 mm에서 중립자세 설정 (최대 회전)", "", 18, Pt(8))

div = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.85), Inches(3.6), Pt(2), Inches(5.4))
div.fill.solid()
div.fill.fore_color.rgb = BLUE
div.line.fill.background()

rtb = _tb(sl, rx, zone_top, col_w, Inches(6.5))
rtf = rtb.text_frame
rtf.word_wrap = True
p = rtf.paragraphs[0]
p.text = "Logged Signals"
p.font.size = Pt(26)
p.font.color.rgb = BLUE
p.font.bold = True
for en in [
    "X-Plane attitude",
    "PX4 자세 추정값",
    "플랫폼 pose command",
    "ACK timing + host timestamps",
]:
    _bullet(rtf, en, None, 20, Pt(10))

_p(rtf, "", 8, sp=Pt(14))
_p(rtf, "Extracted Metrics", 24, BLUE, True, Pt(6))
for en in [
    "Tracking error (raw)",
    "Bias-corrected tracking error",
    "Sample age",
    "Serial RTT / end-to-end delay",
]:
    _bullet(rtf, en, None, 20, Pt(8))

_sn(sl, 6)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 7 — Result 1: Tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "Result 1: Closed-Loop Tracking Is Achieved")

fig2 = os.path.join(FIG, "Fig2_PitchResponse.png")
_img(sl, fig2, Inches(1.0), Inches(3.4), width=Inches(11.5))

atb = _tb(sl, Inches(13.0), Inches(3.6), Inches(5.5), Inches(2.0))
atf = atb.text_frame
atf.word_wrap = True
p = atf.paragraphs[0]
p.text = ""

def _run(txt, bold=False, color=DARK):
    r = p.add_run()
    r.text = txt
    r.font.size = Pt(20)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = FONT_NAME
    return r

orange = RGBColor(0xE6, 0x7E, 0x22)
blue_dashed = RGBColor(0x4A, 0x90, 0xE2)

_run("점선", bold=True)
_run(": 스텝 입력. ")
_run("주황선", bold=True, color=orange)
_run(": FC pitch.\n")
_run("파란 점선", bold=True, color=blue_dashed)
_run(": X-Plane pitch.\n")
_run("파랑선", bold=True, color=BLUE)
_run(": +1.82° 보정된 X-Plane pitch이며 ")
_run("FC pitch", bold=True, color=orange)
_run("와 거의 일치함")

_p(atf, "", 8, sp=Pt(16))
_p(atf, "PX4는 일관된 위상 지연을 보이며 스텝을 추종하는 결과를 보임.", 20, DARK, sp=Pt(4))
_ko(atf, "", 14, Pt(8))

_p(atf, "", 8, sp=Pt(16))
_p(atf, "잔여 오프셋은 설명 가능한 바이어스임. 실험을 반복해도 일관되게 관찰됨.", 20, DARK, sp=Pt(4))
_ko(atf, "", 14, Pt(8))

_callout(sl, Inches(2.67), Inches(8.8), Inches(14.66), Inches(1.2),
         "플랫폼은 정량적 폐루프 검증이 가능할 정도로, 명령된 자세 전이를 충분히 정확하게 재현한다.", 20)

_sn(sl, 7)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 8 — Result 2: Error Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "Result 2: Residual Error Is Interpretable")

pw = Inches(8.0)
fig3 = os.path.join(FIG, "Fig3_PitchError.png")
fig4 = os.path.join(FIG, "Fig4_MountingBias_TimeSeries.png")
_img(sl, fig3, Inches(1.0), Inches(3.4), width=pw)
_img(sl, fig4, Inches(10.0), Inches(3.4), width=pw)

cap1 = _tb(sl, Inches(1.0), Inches(7.6), pw, Inches(0.5))
p = cap1.text_frame.paragraphs[0]
p.text = "바이어스 보정을 적용하면 오차 분포가 0에 더 가까워진다."
p.font.size = Pt(16)
p.font.color.rgb = GREY
p.font.italic = True
p.alignment = PP_ALIGN.CENTER

cap2 = _tb(sl, Inches(10.0), Inches(7.6), pw, Inches(0.5))
p = cap2.text_frame.paragraphs[0]
p.text = "단단히 고정한 실험에서는 장착 바이어스와 일치하는 안정적인 오프셋이 관찰된다."
p.font.size = Pt(16)
p.font.color.rgb = GREY
p.font.italic = True
p.alignment = PP_ALIGN.CENTER

_callout(sl, Inches(2.67), Inches(8.8), Inches(14.66), Inches(1.2), "지배적인 잔여 오차는 무작위 불안정이 아니며, 고정 바이어스는 동적 추종 충실도와 분리해 해석해야 한다.", 20)

_sn(sl, 8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 9 — Result 3: Latency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "Result 3: Latency Sets the Fidelity Limit")

fig5a = os.path.join(FIG, "Fig5a_Latency_TimeSeries.png")
fig5b = os.path.join(FIG, "Fig5b_Latency_Distribution_Baseline.png")
_img(sl, fig5a, Inches(1.0), Inches(3.4), width=Inches(8.5))
_img(sl, fig5b, Inches(10.0), Inches(3.4), width=Inches(8.5))

chip_top = Inches(7.5)
chip_h = Inches(0.6)
chip_w = Inches(4.5)
chips = [
    ("Serial RTT mean \u2248 22 ms", Inches(1.5)),
    ("PX4 rx\u2192ACK mean \u2248 41 ms", Inches(7.0)),
    ("X-Plane rx\u2192ACK mean \u2248 51 ms", Inches(13.0)),
]
for text, x in chips:
    ctb = _tb(sl, x, chip_top, chip_w, chip_h)
    p = ctb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(18)
    p.font.color.rgb = BLUE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

ctf = _callout(sl, Inches(2.67), Inches(8.8), Inches(14.66), Inches(1.2),"측정된 지연과 지터가 위상 지연을 설명하며, 이 플랫폼의 실질적 대역폭을 정의한다.", 20)

_sn(sl, 9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 10 — Takeaways
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "Takeaways")

col_w = Inches(7.5)
col_top = Inches(3.4)
lx = Inches(1.8)
rx = Inches(10.7)

ltb = _tb(sl, lx, col_top, col_w, Inches(5.5))
ltf = ltb.text_frame
ltf.word_wrap = True
p = ltf.paragraphs[0]
p.text = "Key Results"
p.font.size = Pt(26)
p.font.color.rgb = BLUE
p.font.bold = True

takeaways = [
    ("통합: X-Plane + Stewart + PX4를\n하나의 물리적 폐루프 구조로 통합",
     "통합: 물리적 폐루프 HILS 아키텍처 구현"),
    ("충실도: 폐루프 자세 추종 달성 및 정량적 평가",
     "충실도: 폐루프 자세 추종 달성 및 정량 평가"),
    ("범용성: 기체에 종속되지 않는 시스템.\n고정익기, 멀티로터, 인공위성 등 다양한 기체에 적용할수 있는 testbed 개발",
     "범용성: 기체 비종속 구조 \u2014 동일 환경으로 다양한 기체 검증"),
]
for i, (en, kr) in enumerate(takeaways):
    _p(ltf, f"{i + 1}.  {en}", 20, DARK, sp=Pt(14))

div = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.85), Inches(3.5), Pt(2), Inches(5.0))
div.fill.solid()
div.fill.fore_color.rgb = BLUE
div.line.fill.background()

rtb = _tb(sl, rx, col_top, col_w, Inches(5.5))
rtf = rtb.text_frame
rtf.word_wrap = True
p = rtf.paragraphs[0]
p.text = "Next Steps"
p.font.size = Pt(26)
p.font.color.rgb = BLUE
p.font.bold = True

for en, kr in [
    ("시간 정렬후 오차 분석", "시간 정렬 기반 오차 분석"),
    ("반복성·히스테리시스 특성 분석",
     "반복성·히스테리시스 특성 분석"),
    ("작동영역 및 포화 한계 부근 스트레스 테스트",
     "작업 영역·포화 한계 스트레스 시험"),
    ("다양한 기체에 대한 검증 진행 (멀티로터, 위성 등)",
     "다기체 검증 (멀티로터, 위성 등)"),
]:
    _p(rtf, f"\u2192  {en}", 20, DARK, sp=Pt(14))

_callout(sl, Inches(2.67), Inches(8.8), Inches(14.66), Inches(1.2),
         "소프트웨어 전용 HILS와 실제 비행 시험 사이의 격차를 메우는 물리적 검증 수단", 20)

_sn(sl, 10)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 11 — End / Thank You
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)

tb = _tb(sl, Inches(2), Inches(3.0), Inches(16), Inches(1.5))
p = tb.text_frame.paragraphs[0]
p.text = "Thank You"
p.font.size = Pt(56)
p.font.color.rgb = BLACK
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

tb2 = _tb(sl, Inches(2), Inches(5.0), Inches(16), Inches(1.0))
p = tb2.text_frame.paragraphs[0]
p.text = "감사합니다"
p.font.size = Pt(32)
p.font.color.rgb = KO_GREY
p.alignment = PP_ALIGN.CENTER

tb3 = _tb(sl, Inches(2), Inches(6.5), Inches(16), Inches(1.0))
p = tb3.text_frame.paragraphs[0]
p.text = "Questions welcome"
p.font.size = Pt(24)
p.font.color.rgb = BLUE
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

_sn(sl, 11)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 12 — Backup: z=+20 mm Proof
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "Q&A")

rot_fig = os.path.join(FIG, "rot_worst_vs_z.png")
_img(sl, rot_fig, Inches(2.5), Inches(3.4), width=Inches(10.0))

ann = _tb(sl, Inches(13.5), Inches(3.6), Inches(5.5), Inches(4.0))
atf = ann.text_frame
atf.word_wrap = True
p = atf.paragraphs[0]
p.text = "z = +20 mm"
p.font.size = Pt(28)
p.font.color.rgb = BLUE
p.font.bold = True
_p(atf, "\u2192 ~20\u00b0 rotation range", 22, DARK, sp=Pt(8))
_p(atf, "Sufficient for \u00b15\u00b0 trim test\nwith large margin.", 18, DARK, sp=Pt(16))
_p(atf, "", 8, sp=Pt(24))
_p(atf, "z = 0 mm", 28, RED, True, Pt(8))
_p(atf, "\u2192 ~10\u00b0 rotation range", 22, DARK, sp=Pt(8))
_p(atf, "Insufficient for \u00b15\u00b0 test.", 18, GREY, sp=Pt(16))

_sn(sl, 12)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 13 — Backup: Anticipated Questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = make_slide(prs, blank)
slide_title(sl, "Q&A")

questions = [
    ("Why not conventional HILS?",
     "Real inertial excitation that sensor\nsynthesis cannot provide. \u2192 Sl 3"),
    ("How do you handle saturation?",
     "Baseline \u00b15\u00b0 out of ~20\u00b0 available.\nStress tests planned. \u2192 Sl 6+10"),
    ("Effective bandwidth?",
     "~50 Hz tick, E2E latency ~51 ms.\nLow-freq maneuvers suited. \u2192 Sl 5+9"),
    ("Open-loop platform \u2014 pose accuracy?",
     "Hobby servos, no encoder. Inclinometer:\n~2.1\u00b0 roll, ~1.5\u00b0 pitch. \u2192 Sl 3"),
    ("What is mounting bias?",
     "~1.8\u00b0 from FC alignment. Stable\n(\u03c3 \u2248 0.2\u00b0), calibrated out. \u2192 Sl 8"),
    ("How repeatable across runs?",
     "5 runs: \u03c3 \u2248 0.2\u00b0 intra, \u0394 \u2264 0.8\u00b0 inter.\n\u2192 Sl 8+10"),
    ("What about yaw?",
     "In the loop, but trim scenario\nexcites pitch primarily. \u2192 Sl 6"),
    ("Extend to quadrotors?",
     "Vehicle-agnostic architecture.\nOnly model + PID change. \u2192 Sl 10"),
]

cols = 4
rows = 2
cw = Inches(4.2)
ch = Inches(2.6)
cgap_x = Inches(0.3)
cgap_y = Inches(0.3)
total_w = cols * cw + (cols - 1) * cgap_x
grid_left = (prs.slide_width - total_w) // 2
grid_top = Inches(3.3)

for i, (q, a) in enumerate(questions):
    col = i % cols
    row = i // cols
    x = grid_left + col * (cw + cgap_x)
    y = grid_top + row * (ch + cgap_y)

    _card(sl, x, y, cw, ch)

    qtb = _tb(sl, x + Inches(0.25), y + Inches(0.2), cw - Inches(0.5), Inches(0.6))
    p = qtb.text_frame.paragraphs[0]
    p.text = q
    p.font.size = Pt(16)
    p.font.color.rgb = BLUE
    p.font.bold = True

    atb = _tb(sl, x + Inches(0.25), y + Inches(0.85), cw - Inches(0.5), ch - Inches(1.1))
    atf = atb.text_frame
    atf.word_wrap = True
    p = atf.paragraphs[0]
    p.text = a
    p.font.size = Pt(14)
    p.font.color.rgb = DARK
    p.line_spacing = Pt(20)

_sn(sl, 13)


# ── Save ────────────────────────────────────────────────────────────

out_path = os.path.join(BASE, "KSAS_Spring2026_Presentation.pptx")
prs.save(out_path)
print(f"Saved: {out_path}")
