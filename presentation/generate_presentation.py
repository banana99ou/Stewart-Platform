"""Generate KSAS Spring Conference presentation for Stewart Platform HILS."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "fig")

ACCENT = RGBColor(0x1B, 0x6B, 0x93)
ACCENT_LIGHT = RGBColor(0xD6, 0xEE, 0xF5)
WARN = RGBColor(0xC0, 0x39, 0x2B)
WARN_LIGHT = RGBColor(0xFA, 0xE5, 0xE3)
DARK = RGBColor(0x21, 0x21, 0x21)
MEDIUM = RGBColor(0x55, 0x55, 0x55)
KO_COLOR = RGBColor(0x99, 0x99, 0x99)
LIGHT_BG = RGBColor(0xF5, 0xF5, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TOTAL = 12
W = Inches(13.333)
H = Inches(7.5)
MARGIN_L = Inches(0.7)
MARGIN_R = Inches(0.7)
CONTENT_W = Inches(13.333 - 1.4)


def add_rounded_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def set_tf(shape, text, font_size=12, color=DARK, bold=False, alignment=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = alignment
    return tf


def add_para(tf, text, font_size=12, color=DARK, bold=False, space_before=Pt(4), alignment=PP_ALIGN.LEFT):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.space_before = space_before
    p.alignment = alignment
    return p


def ko(tf, text, font_size=10, space_before=Pt(1), alignment=PP_ALIGN.LEFT):
    """Add a Korean translation line in grey."""
    return add_para(tf, text, font_size, KO_COLOR, False, space_before, alignment)


def add_title(slide, en, kr=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.15))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.word_wrap = True
    tf.margin_left = MARGIN_L
    tf.margin_top = Pt(14)
    tf.margin_bottom = Pt(4)
    p = tf.paragraphs[0]
    p.text = en
    p.font.size = Pt(28)
    p.font.color.rgb = WHITE
    p.font.bold = True
    if kr:
        p2 = tf.add_paragraph()
        p2.text = kr
        p2.font.size = Pt(15)
        p2.font.color.rgb = RGBColor(0xA8, 0xD4, 0xE6)
        p2.space_before = Pt(2)
    return bar


def add_bottom_bar(slide, text, color=ACCENT, text_color=WHITE, top=Inches(6.55)):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, top, W, Inches(0.85))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.word_wrap = True
    tf.margin_left = MARGIN_L
    tf.margin_right = MARGIN_R
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(16)
    p.font.color.rgb = text_color
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(6)
    return bar


def add_image_safe(slide, path, left, top, width=None, height=None):
    if os.path.exists(path):
        return slide.shapes.add_picture(path, left, top, width, height)
    box = add_rounded_rect(slide, left, top, width or Inches(5), height or Inches(3), LIGHT_BG, MEDIUM)
    set_tf(box, f"[Image not found: {os.path.basename(path)}]", 11, MEDIUM, alignment=PP_ALIGN.CENTER)
    return box


def sn(slide, num):
    tb = slide.shapes.add_textbox(Inches(12.5), Inches(7.1), Inches(0.7), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = f"{num}/{TOTAL}"
    p.font.size = Pt(10)
    p.font.color.rgb = MEDIUM
    p.alignment = PP_ALIGN.RIGHT


def bullet(tf, en, kr_=None, en_size=14, space=Pt(8)):
    add_para(tf, f"•  {en}", en_size, DARK, space_before=space)
    if kr_:
        ko(tf, f"    {kr_}", en_size - 3, Pt(1))


# ── Build presentation ──────────────────────────────────────────────

prs = Presentation()
prs.slide_width = W
prs.slide_height = H
blank = prs.slide_layouts[6]

# ━━ SLIDE 1: Title ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE

hero_bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(2.6))
hero_bar.fill.solid()
hero_bar.fill.fore_color.rgb = ACCENT
hero_bar.line.fill.background()
hero_bar.shadow.inherit = False

tf = hero_bar.text_frame
tf.word_wrap = True
tf.margin_left = MARGIN_L
tf.margin_top = Inches(0.4)
p = tf.paragraphs[0]
p.text = "Physically Closed-Loop HILS for\nFlight Attitude-Control Validation"
p.font.size = Pt(32)
p.font.color.rgb = WHITE
p.font.bold = True
p.line_spacing = Pt(40)

add_para(tf, "Using a Stewart Platform with X-Plane and PX4", 18, ACCENT_LIGHT, space_before=Pt(10))
ko(tf, "스튜어트 플랫폼 · X-Plane · PX4 연동 기반 비행체 자세제어 검증", 13, Pt(2))

claim_box = add_rounded_rect(sl, MARGIN_L, Inches(2.95), Inches(7.5), Inches(1.3), ACCENT_LIGHT)
claim_box.text_frame.word_wrap = True
claim_box.text_frame.margin_left = Pt(14)
claim_box.text_frame.margin_top = Pt(10)
claim_box.text_frame.margin_right = Pt(14)
set_tf(claim_box,
    "X-Plane drives real platform motion \u2192 PX4 senses that motion physically \u2192 "
    "measured attitude closes the loop.",
    14, ACCENT, bold=True)
ko(claim_box.text_frame,
    "X-Plane \u2192 실제 모션 구동 \u2192 PX4 물리적 감지 \u2192 측정 자세로 폐루프 완성", 11, Pt(4))

fig_path = os.path.join(FIG, "Fig1_Architecture.png")
add_image_safe(sl, fig_path, Inches(8.5), Inches(2.85), width=Inches(4.3))

badge = add_rounded_rect(sl, MARGIN_L, Inches(4.6), Inches(3.8), Inches(0.45), WARN_LIGHT)
set_tf(badge, "Beyond sensor-synthesis-only HILS", 12, WARN, bold=True, alignment=PP_ALIGN.CENTER)
badge.text_frame.margin_top = Pt(4)

author_tb = sl.shapes.add_textbox(MARGIN_L, Inches(6.4), Inches(10), Inches(0.9))
atf = author_tb.text_frame
atf.word_wrap = True
add_para(atf, "Spring 2026 KSAS Conference", 14, MEDIUM, bold=True, space_before=Pt(0))
ko(atf, "2026 춘계 항공우주학회", 11)
add_para(atf, "[Author Name] \u2014 [Affiliation]", 12, MEDIUM, space_before=Pt(4))

sn(sl, 1)

# ━━ SLIDE 2: Motivation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "Why Add Physical Motion to HILS?", "왜 HILS에 물리적 모션이 필요한가?")

col_w = Inches(5.5)
col_h = Inches(3.8)
left_x = MARGIN_L + Inches(0.2)
right_x = Inches(7.1)
col_top = Inches(1.55)

left_box = add_rounded_rect(sl, left_x, col_top, col_w, col_h, LIGHT_BG, MEDIUM)
ltf = left_box.text_frame
ltf.word_wrap = True
ltf.margin_left = Pt(16)
ltf.margin_top = Pt(14)
ltf.margin_right = Pt(12)
p = ltf.paragraphs[0]
p.text = "Conventional HILS"
p.font.size = Pt(20)
p.font.color.rgb = MEDIUM
p.font.bold = True
ko(ltf, "기존 HILS", 12)
for en, kr_ in [
    ("Simulator computes dynamics", "시뮬레이터가 동역학 계산"),
    ("Controller receives synthetic sensor signals", "합성 센서 신호 수신"),
    ("Attitude evaluated via logs / screens", "로그/화면으로만 자세 확인"),
    ("IMU does not experience real inertial motion", "IMU가 실제 관성 운동 미경험"),
]:
    bullet(ltf, en, kr_, 14, Pt(8))

right_box = add_rounded_rect(sl, right_x, col_top, col_w, col_h, ACCENT_LIGHT, ACCENT)
rtf = right_box.text_frame
rtf.word_wrap = True
rtf.margin_left = Pt(16)
rtf.margin_top = Pt(14)
rtf.margin_right = Pt(12)
p = rtf.paragraphs[0]
p.text = "This Work"
p.font.size = Pt(20)
p.font.color.rgb = ACCENT
p.font.bold = True
ko(rtf, "본 연구", 12)
for en, kr_ in [
    ("Simulator drives real Stewart platform motion", "실제 플랫폼 모션 구동"),
    ("PX4 IMU senses real inertial excitation", "PX4 IMU 실제 관성 자극 감지"),
    ("Measured attitude fed back into control loop", "측정 자세 → 제어 루프 피드백"),
    ("Timing and fidelity quantitatively measured", "타이밍·충실도 정량 측정"),
]:
    bullet(rtf, en, kr_, 14, Pt(8))

add_bottom_bar(sl,
    "Can a physically closed loop remain stable, repeatable, and quantitatively explainable?",
    WARN, WHITE, top=Inches(5.7))

sn(sl, 2)

# ━━ SLIDE 3: What Is New ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "What Is New Here?", "본 연구의 차별점")

central = add_rounded_rect(sl, MARGIN_L, Inches(1.5), CONTENT_W, Inches(1.15), ACCENT_LIGHT)
central.text_frame.word_wrap = True
central.text_frame.margin_left = Pt(18)
central.text_frame.margin_top = Pt(10)
set_tf(central,
    "Closed-loop feedback uses attitude estimated from real IMU motion, "
    "not only simulated sensor values.",
    18, ACCENT, bold=True, alignment=PP_ALIGN.CENTER)
ko(central.text_frame,
    "폐루프 피드백이 실제 IMU 운동으로부터 추정된 자세를 사용", 12, Pt(4), PP_ALIGN.CENTER)

block_w = Inches(3.75)
block_h = Inches(2.1)
block_top = Inches(3.0)
gap = Inches(0.35)
block_x = [MARGIN_L + Inches(0.15), MARGIN_L + Inches(0.15) + block_w + gap,
           MARGIN_L + Inches(0.15) + 2*(block_w + gap)]

labels = [
    ("Realized Motion", "모션 구현",
     "Simulator attitude is converted into\nStewart-platform motion.",
     "시뮬레이터 자세 → 플랫폼 물리 모션"),
    ("Physical Sensing", "물리적 센싱",
     "PX4 IMU and estimator respond to\nreal inertial excitation.",
     "PX4 IMU가 실제 관성 자극에 반응"),
    ("Closed-Loop Return", "폐루프 귀환",
     "Measured attitude is sent back into\nthe host control loop.",
     "측정 자세가 호스트 제어 루프로 귀환"),
]

for i, (title_en, title_kr, desc_en, desc_kr) in enumerate(labels):
    box = add_rounded_rect(sl, block_x[i], block_top, block_w, block_h, LIGHT_BG)
    btf = box.text_frame
    btf.word_wrap = True
    btf.margin_left = Pt(14)
    btf.margin_top = Pt(14)
    btf.margin_right = Pt(14)
    p = btf.paragraphs[0]
    p.text = title_en
    p.font.size = Pt(18)
    p.font.color.rgb = ACCENT
    p.font.bold = True
    ko(btf, title_kr, 11)
    add_para(btf, desc_en, 14, DARK, space_before=Pt(8))
    ko(btf, desc_kr, 10)

    num_circle = sl.shapes.add_shape(MSO_SHAPE.OVAL, block_x[i] + block_w - Inches(0.55),
                                      block_top + Inches(0.12), Inches(0.42), Inches(0.42))
    num_circle.fill.solid()
    num_circle.fill.fore_color.rgb = ACCENT
    num_circle.line.fill.background()
    num_circle.shadow.inherit = False
    set_tf(num_circle, str(i+1), 16, WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    num_circle.text_frame.margin_top = Pt(2)

caution = add_rounded_rect(sl, MARGIN_L, Inches(5.5), CONTENT_W, Inches(0.7), WARN_LIGHT, WARN)
caution.text_frame.word_wrap = True
caution.text_frame.margin_left = Pt(14)
caution.text_frame.margin_top = Pt(6)
set_tf(caution,
    "This is a complementary validation layer, not a full aerodynamic flight replacement.",
    14, WARN, bold=True, alignment=PP_ALIGN.CENTER)
ko(caution.text_frame,
    "보완적 검증 계층이며, 공력 비행을 완전히 대체하지 않음", 11, Pt(2), PP_ALIGN.CENTER)

sn(sl, 3)

# ━━ SLIDE 4: System in Action ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "System in Action", "시스템 구동 장면")

photo_w = Inches(5.5)
photo_h = Inches(4.0)
photo_box = add_rounded_rect(sl, Inches(0.8), Inches(1.6), photo_w, photo_h, LIGHT_BG, MEDIUM)
photo_box.text_frame.word_wrap = True
photo_box.text_frame.margin_top = Inches(1.4)
photo_box.text_frame.margin_left = Pt(14)
set_tf(photo_box, "[Insert hardware photo]", 16, MEDIUM, alignment=PP_ALIGN.CENTER)
ko(photo_box.text_frame, "스튜어트 플랫폼 + PX4 탑재 사진", 12, Pt(6), PP_ALIGN.CENTER)

video_box = add_rounded_rect(sl, Inches(6.9), Inches(1.6), Inches(5.5), Inches(4.0), LIGHT_BG, MEDIUM)
video_box.text_frame.word_wrap = True
video_box.text_frame.margin_top = Inches(1.4)
video_box.text_frame.margin_left = Pt(14)
set_tf(video_box, "[Insert demo video]", 16, MEDIUM, alignment=PP_ALIGN.CENTER)
ko(video_box.text_frame, "Trim transition 시나리오 구동 영상 (5\u20138 s)", 12, Pt(6), PP_ALIGN.CENTER)

desc_tb = sl.shapes.add_textbox(MARGIN_L, Inches(5.9), CONTENT_W, Inches(1.0))
dtf = desc_tb.text_frame
dtf.word_wrap = True
p = dtf.paragraphs[0]
p.text = ("X-Plane sim attitude \u2192 Stewart platform physical motion \u2192 "
          "PX4 IMU sensing \u2192 closed-loop control")
p.font.size = Pt(14)
p.font.color.rgb = ACCENT
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

sn(sl, 4)

# ━━ SLIDE 5: System Architecture ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "How the Physical Closed Loop Works", "물리적 폐루프의 작동 원리")

fig1 = os.path.join(FIG, "Fig1_Architecture.png")
IMG_W_PX, IMG_H_PX = 1646, 1122
fig_left = Inches(0.5)
fig_top = Inches(1.35)
fig_w = Inches(7.8)
add_image_safe(sl, fig1, fig_left, fig_top, width=fig_w)

px_per_inch = IMG_W_PX / 7.8
diagram_circles = [
    ("1", 170,  240),
    ("2", 100,  580),
    ("3", 700,  850),
    ("4", 1480, 560),
]
circle_d = Inches(0.38)
for num, px_x, px_y in diagram_circles:
    cx = fig_left + Inches(px_x / px_per_inch) - circle_d / 2
    cy = fig_top + Inches(px_y / px_per_inch) - circle_d / 2
    circ = sl.shapes.add_shape(MSO_SHAPE.OVAL, cx, cy, circle_d, circle_d)
    circ.fill.solid()
    circ.fill.fore_color.rgb = WARN
    circ.line.color.rgb = WHITE
    circ.line.width = Pt(2)
    circ.shadow.inherit = False
    set_tf(circ, num, 14, WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    circ.text_frame.margin_top = Pt(2)
    circ.text_frame.margin_left = Pt(0)
    circ.text_frame.margin_right = Pt(0)

callout_x = Inches(8.7)
callout_top = Inches(1.5)

steps = [
    ("1", "X-Plane publishes vehicle state\nto the Python host over UDP.",
          "X-Plane → UDP → Python 호스트"),
    ("2", "Host converts state to pose\ncommands for the Stewart platform.",
          "호스트 → pose 명령 → 스튜어트 플랫폼"),
    ("3", "PX4 physically senses platform\nmotion through its onboard IMU.",
          "PX4 IMU가 플랫폼 모션을 물리적으로 감지"),
    ("4", "PX4 attitude returns to host;\nhost injects control into X-Plane.",
          "PX4 자세 → 호스트 → X-Plane 제어 주입"),
]
for i, (num, en, kr_) in enumerate(steps):
    y = callout_top + Inches(i * 1.25)
    circ = sl.shapes.add_shape(MSO_SHAPE.OVAL, callout_x, y, Inches(0.4), Inches(0.4))
    circ.fill.solid()
    circ.fill.fore_color.rgb = ACCENT
    circ.line.fill.background()
    circ.shadow.inherit = False
    set_tf(circ, num, 14, WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    circ.text_frame.margin_top = Pt(2)

    tb = sl.shapes.add_textbox(callout_x + Inches(0.55), y - Pt(2), Inches(3.8), Inches(1.0))
    ttf = tb.text_frame
    ttf.word_wrap = True
    p = ttf.paragraphs[0]
    p.text = en
    p.font.size = Pt(13)
    p.font.color.rgb = DARK
    p.line_spacing = Pt(18)
    ko(ttf, kr_, 10, Pt(2))

add_bottom_bar(sl,
    "The loop is closed across software, mechanics, sensing, and control I/O.",
    ACCENT, WHITE, top=Inches(6.55))

sn(sl, 5)

# ━━ SLIDE 6: Experiment Design ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "What We Tested and How We Measured It", "실험 시나리오 및 측정 방법")

left_w = Inches(5.6)
right_w = Inches(5.6)
zone_top = Inches(1.55)
zone_h = Inches(4.1)

left_box = add_rounded_rect(sl, MARGIN_L, zone_top, left_w, zone_h, LIGHT_BG)
ltf = left_box.text_frame
ltf.word_wrap = True
ltf.margin_left = Pt(18)
ltf.margin_top = Pt(14)
ltf.margin_right = Pt(14)
p = ltf.paragraphs[0]
p.text = "Baseline Scenario"
p.font.size = Pt(20)
p.font.color.rgb = ACCENT
p.font.bold = True
ko(ltf, "기본 실험 시나리오", 12)
add_para(ltf, "Pitch trim transition:", 15, DARK, bold=True, space_before=Pt(14))
add_para(ltf, "0\u00b0 \u2192 +5\u00b0 \u2192 hold \u2192 0\u00b0", 18, ACCENT, bold=True, space_before=Pt(4))
add_para(ltf, "", 6, space_before=Pt(6))
bullet(ltf, "Each run starts from the same\nsimulator initial condition",
       "매 실험 동일 초기 조건", 13, Pt(6))
bullet(ltf, "Repeated multiple times for statistics",
       "통계 확보를 위해 반복 수행", 13, Pt(6))
bullet(ltf, "Platform neutral at z=+20 mm\nto maximize rotation range",
       "회전 범위 극대화를 위해 z=+20 mm 기본 자세", 13, Pt(6))
bullet(ltf, "Baseline runs stay inside workspace\n(nominal behavior, not saturation)",
       "작업 영역 내 정상 거동", 13, Pt(6))

right_box = add_rounded_rect(sl, Inches(6.9), zone_top, right_w, zone_h, LIGHT_BG)
rtf = right_box.text_frame
rtf.word_wrap = True
rtf.margin_left = Pt(18)
rtf.margin_top = Pt(14)
rtf.margin_right = Pt(14)
p = rtf.paragraphs[0]
p.text = "Logged Signals"
p.font.size = Pt(20)
p.font.color.rgb = ACCENT
p.font.bold = True
ko(rtf, "기록 신호", 12)

for en, kr_ in [
    ("X-Plane attitude", "X-Plane 자세"),
    ("PX4 attitude estimate", "PX4 자세 추정값"),
    ("Platform pose command", "플랫폼 pose 명령"),
    ("ACK timing and host timestamps", "ACK 타이밍 및 타임스탬프"),
]:
    bullet(rtf, en, kr_, 14, Pt(6))

add_para(rtf, "", 6, space_before=Pt(8))
add_para(rtf, "Extracted Metrics", 17, ACCENT, bold=True, space_before=Pt(4))
ko(rtf, "추출 메트릭", 11)
for en, kr_ in [
    ("Tracking error (raw)", "추종 오차 (원시)"),
    ("Bias-corrected tracking error", "바이어스 보정 추종 오차"),
    ("Sample age", None),
    ("Serial RTT / end-to-end delay", "종단간 지연"),
]:
    bullet(rtf, en, kr_, 14, Pt(5))

sn(sl, 6)

# ━━ SLIDE 7: Tracking Result ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "Result 1: Closed-Loop Tracking Is Achieved", "결과 1: 폐루프 자세 추종 달성")

fig2 = os.path.join(FIG, "Fig2_PitchResponse.png")
add_image_safe(sl, fig2, Inches(0.8), Inches(1.35), width=Inches(8.5))

ann1 = add_rounded_rect(sl, Inches(9.6), Inches(1.8), Inches(3.3), Inches(1.3), ACCENT_LIGHT)
ann1.text_frame.word_wrap = True
ann1.text_frame.margin_left = Pt(12)
ann1.text_frame.margin_top = Pt(10)
set_tf(ann1, "PX4 follows the step response\nwith consistent phase lag.", 14, ACCENT)
ko(ann1.text_frame, "일관된 위상 지연으로 스텝 응답 추종", 10, Pt(6))

ann2 = add_rounded_rect(sl, Inches(9.6), Inches(3.4), Inches(3.3), Inches(1.3), ACCENT_LIGHT)
ann2.text_frame.word_wrap = True
ann2.text_frame.margin_left = Pt(12)
ann2.text_frame.margin_top = Pt(10)
set_tf(ann2, "Residual offset is structured \u2014\nexplainable bias, not instability.", 14, ACCENT)
ko(ann2.text_frame, "잔여 오프셋은 불안정이 아닌 설명 가능한 바이어스", 10, Pt(6))

takeaway = add_rounded_rect(sl, MARGIN_L, Inches(5.8), CONTENT_W, Inches(0.8), ACCENT)
takeaway.text_frame.word_wrap = True
takeaway.text_frame.margin_left = Pt(18)
takeaway.text_frame.margin_top = Pt(8)
set_tf(takeaway,
    "The platform reproduces commanded attitude transitions well enough "
    "for quantitative closed-loop validation.",
    15, WHITE, bold=True, alignment=PP_ALIGN.CENTER)

sn(sl, 7)

# ━━ SLIDE 8: Error Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "Result 2: Residual Error Is Interpretable", "결과 2: 잔여 오차의 해석")

fig3 = os.path.join(FIG, "Fig3_PitchError.png")
fig4 = os.path.join(FIG, "Fig4_MountingBias_TimeSeries.png")
panel_w = Inches(5.8)
add_image_safe(sl, fig3, Inches(0.5), Inches(1.5), width=panel_w)
add_image_safe(sl, fig4, Inches(6.7), Inches(1.5), width=panel_w)

cap1 = sl.shapes.add_textbox(Inches(0.5), Inches(5.0), panel_w, Inches(0.5))
p = cap1.text_frame.paragraphs[0]
p.text = "Bias correction moves the error distribution closer to zero."
p.font.size = Pt(12)
p.font.color.rgb = MEDIUM
p.font.italic = True
p.alignment = PP_ALIGN.CENTER

cap2 = sl.shapes.add_textbox(Inches(6.7), Inches(5.0), panel_w, Inches(0.5))
p = cap2.text_frame.paragraphs[0]
p.text = "Firm-mounted run shows stable offset consistent with mounting bias."
p.font.size = Pt(12)
p.font.color.rgb = MEDIUM
p.font.italic = True
p.alignment = PP_ALIGN.CENTER

add_bottom_bar(sl,
    "The dominant residual is not random instability \u2014 "
    "a fixed bias must be separated from dynamic tracking fidelity.",
    WARN, WHITE, top=Inches(5.7))

sn(sl, 8)

# ━━ SLIDE 9: Latency ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "Result 3: Latency Sets the Fidelity Limit", "결과 3: 지연 시간이 충실도의 한계를 결정")

fig5a = os.path.join(FIG, "Fig5a_Latency_TimeSeries.png")
fig5b = os.path.join(FIG, "Fig5b_Latency_Distribution_Baseline.png")
add_image_safe(sl, fig5a, Inches(0.4), Inches(1.5), width=Inches(6.2))
add_image_safe(sl, fig5b, Inches(6.8), Inches(1.5), width=Inches(5.8))

chip_top = Inches(5.1)
chip_h = Inches(0.42)
chip_w = Inches(3.3)
chips = [
    ("Serial RTT mean \u2248 22 ms", Inches(0.7)),
    ("PX4 rx\u2192ACK mean \u2248 41 ms", Inches(4.3)),
    ("X-Plane rx\u2192ACK mean \u2248 51 ms", Inches(8.5)),
]
for text, x in chips:
    chip = add_rounded_rect(sl, x, chip_top, chip_w, chip_h, ACCENT_LIGHT)
    chip.text_frame.word_wrap = True
    chip.text_frame.margin_left = Pt(8)
    chip.text_frame.margin_top = Pt(4)
    set_tf(chip, text, 12, ACCENT, bold=True, alignment=PP_ALIGN.CENTER)

add_bottom_bar(sl,
    "Measured delay and jitter explain the phase lag and define "
    "the practical bandwidth of this platform.",
    ACCENT, WHITE, top=Inches(5.85))

note_tb = sl.shapes.add_textbox(MARGIN_L, Inches(6.85), CONTENT_W, Inches(0.35))
p = note_tb.text_frame.paragraphs[0]
p.text = "Baseline runs show near-zero saturation \u2192 nominal timing dominates."
p.font.size = Pt(11)
p.font.color.rgb = MEDIUM
p.font.italic = True
p.alignment = PP_ALIGN.CENTER

sn(sl, 9)

# ━━ SLIDE 10: Limitations ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "What This Platform Does Not Claim", "본 플랫폼이 주장하지 않는 것")

grid_w = Inches(5.6)
grid_h = Inches(2.0)
grid_gap_x = Inches(0.5)
grid_gap_y = Inches(0.35)
grid_left = MARGIN_L + Inches(0.15)
grid_top = Inches(1.55)

limits = [
    ("Finite Workspace", "유한한 작업 영역",
     "Large multi-axis commands reduce\nthe feasible operating region.",
     "대각도 다축 명령 시 구동 영역 축소"),
    ("No Physical Aerodynamic Load", "공력 하중 미재현",
     "The platform reproduces motion cues,\nnot full external-force physics.",
     "모션 큐 재현, 외력 물리 미구현"),
    ("Platform-Level Actuation Limits", "플랫폼 구동 한계",
     "Timing, hysteresis, and actuator constraints\nstill shape response fidelity.",
     "타이밍·히스테리시스가 응답 충실도에 영향"),
    ("Not a Flight Replacement", "비행 대체 불가",
     "This platform complements conventional\nHILS and real flight testing.",
     "기존 HILS 및 실제 비행 시험을 보완"),
]

positions = [
    (grid_left, grid_top),
    (grid_left + grid_w + grid_gap_x, grid_top),
    (grid_left, grid_top + grid_h + grid_gap_y),
    (grid_left + grid_w + grid_gap_x, grid_top + grid_h + grid_gap_y),
]

for (title_en, title_kr, desc_en, desc_kr), (x, y) in zip(limits, positions):
    box = add_rounded_rect(sl, x, y, grid_w, grid_h, WARN_LIGHT, WARN)
    btf = box.text_frame
    btf.word_wrap = True
    btf.margin_left = Pt(18)
    btf.margin_top = Pt(14)
    btf.margin_right = Pt(14)
    p = btf.paragraphs[0]
    p.text = title_en
    p.font.size = Pt(18)
    p.font.color.rgb = WARN
    p.font.bold = True
    ko(btf, title_kr, 11)
    add_para(btf, desc_en, 14, DARK, space_before=Pt(8))
    ko(btf, desc_kr, 10)

add_bottom_bar(sl,
    "Correct claim: a practical, measurable, physically informed pre-flight validation layer.",
    ACCENT, WHITE, top=Inches(6.15))

sn(sl, 10)

# ━━ SLIDE 11: Takeaways ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "Takeaways", "결론 및 향후 과제")

tk_x = MARGIN_L
tk_top = Inches(1.55)
tk_w = Inches(7.0)
tk_h = Inches(4.5)

tk_box = add_rounded_rect(sl, tk_x, tk_top, tk_w, tk_h, ACCENT_LIGHT)
tktf = tk_box.text_frame
tktf.word_wrap = True
tktf.margin_left = Pt(20)
tktf.margin_top = Pt(18)
tktf.margin_right = Pt(14)

p = tktf.paragraphs[0]
p.text = "Key Results"
p.font.size = Pt(20)
p.font.color.rgb = ACCENT
p.font.bold = True
ko(tktf, "주요 성과", 12)

takeaways = [
    ("Physically closed-loop HILS architecture\nimplemented: X-Plane + Stewart + PX4.",
     "물리적 폐루프 HILS 아키텍처 구현"),
    ("Closed-loop attitude tracking achieved\nand evaluated quantitatively.",
     "폐루프 자세 추종 달성 및 정량 평가"),
    ("Bias and latency are measured and used\nto define the platform's true fidelity.",
     "바이어스·지연 측정으로 실제 충실도 정의"),
]
for i, (en, kr_) in enumerate(takeaways):
    add_para(tktf, f"{i+1}.  {en}", 14, DARK, space_before=Pt(12))
    ko(tktf, f"     {kr_}", 10)

ns_x = Inches(8.0)
ns_box = add_rounded_rect(sl, ns_x, tk_top, Inches(4.6), tk_h, LIGHT_BG)
nstf = ns_box.text_frame
nstf.word_wrap = True
nstf.margin_left = Pt(20)
nstf.margin_top = Pt(18)
nstf.margin_right = Pt(14)

p = nstf.paragraphs[0]
p.text = "Next Steps"
p.font.size = Pt(20)
p.font.color.rgb = ACCENT
p.font.bold = True
ko(nstf, "향후 과제", 12)

nexts = [
    ("Time-aligned error analysis", "시간 정렬 기반 오차 분석"),
    ("Repeatability / hysteresis\ncharacterization", "반복성·히스테리시스 특성 분석"),
    ("Stress tests near workspace\nand saturation limits", "작업 영역·포화 한계 스트레스 시험"),
]
for en, kr_ in nexts:
    add_para(nstf, f"\u2192  {en}", 14, DARK, space_before=Pt(12))
    ko(nstf, f"    {kr_}", 10)

add_bottom_bar(sl,
    "This platform fills part of the gap between software-only HILS and real flight testing.",
    ACCENT, WHITE, top=Inches(6.35))

thank = sl.shapes.add_textbox(MARGIN_L, Inches(6.9), CONTENT_W, Inches(0.45))
p = thank.text_frame.paragraphs[0]
p.text = "Thank you \u2014 Questions welcome"
p.font.size = Pt(16)
p.font.color.rgb = ACCENT
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

sn(sl, 11)

# ━━ SLIDE 12: Q&A Backup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sl = prs.slides.add_slide(blank)
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE
add_title(sl, "Backup: Anticipated Questions", "예상 질문 대비")

# Q1: z=+20mm — with proof figure
q1_box = add_rounded_rect(sl, MARGIN_L, Inches(1.45), Inches(5.8), Inches(5.0), LIGHT_BG)
q1tf = q1_box.text_frame
q1tf.word_wrap = True
q1tf.margin_left = Pt(16)
q1tf.margin_top = Pt(12)
q1tf.margin_right = Pt(12)
p = q1tf.paragraphs[0]
p.text = '"Why z=+20 mm as the neutral position?"'
p.font.size = Pt(15)
p.font.color.rgb = ACCENT
p.font.bold = True
ko(q1tf, '"왜 z=+20 mm을 기본 자세로 선택했는가?"', 11)
add_para(q1tf, "", 4, space_before=Pt(4))

rot_fig = os.path.join(FIG, "rot_worst_vs_z.png")
add_image_safe(sl, rot_fig, Inches(1.2), Inches(2.55), width=Inches(4.8))

ann = sl.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(5.2), Inches(0.8))
atf = ann.text_frame
atf.word_wrap = True
p = atf.paragraphs[0]
p.text = "z=+20 mm \u2192 ~20\u00b0 rotation range\nz=0 mm \u2192 ~10\u00b0 (insufficient for \u00b15\u00b0 test with margin)"
p.font.size = Pt(12)
p.font.color.rgb = DARK
p.line_spacing = Pt(17)

# Q2-Q8: compact card grid on the right
questions = [
    ('"Why not conventional HILS?"',
     "Real inertial excitation that sensor synthesis\ncannot provide. → Slides 3 + 10"),

    ('"How do you handle saturation?"',
     "Baseline ±5° out of ~20° available.\nMinimal saturation. Stress tests planned. → Sl 6+10+11"),

    ('"What about yaw?"',
     "In the loop, but trim-transition excites pitch.\nYaw bias handled, not focus here. → Slide 6"),

    ('"Effective bandwidth?"',
     "~50 Hz tick, E2E latency ~51 ms. Suited for\nlow-freq maneuvers; high-freq filtered. → Sl 5+9"),

    ('"Open-loop platform — pose accuracy?"',
     "No encoder feedback on servos. PX4 IMU is\nonly physical measurement. Honest limit. → Sl 10"),

    ('"What is mounting bias physically?"',
     "~1.8° from FC alignment on top plate.\nStable (σ≈0.2°), calibrated in post. → Slide 8"),

    ('"How repeatable across runs?"',
     "5 runs: intra-run σ≈0.2°, inter-run Δ≤0.8°.\nThermal drift + re-seating. → Slides 8+11"),

    ('"Extend to quadrotors?"',
     "Architecture is vehicle-agnostic. Only X-Plane\nmodel + PID change. PX4 supports both. → Sl 11"),
]

col_w = Inches(2.8)
col_gap = Inches(0.15)
col1_x = Inches(6.6)
col2_x = col1_x + col_w + col_gap
card_h = Inches(1.15)
card_gap = Inches(0.12)
ref_top = Inches(1.35)

for i, (q_en, answer) in enumerate(questions):
    col = i % 2
    row = i // 2
    x = col1_x if col == 0 else col2_x
    y = ref_top + row * (card_h + card_gap)
    qbox = add_rounded_rect(sl, x, y, col_w, card_h, LIGHT_BG)
    qtf = qbox.text_frame
    qtf.word_wrap = True
    qtf.margin_left = Pt(8)
    qtf.margin_top = Pt(6)
    qtf.margin_right = Pt(6)
    qtf.margin_bottom = Pt(4)
    p = qtf.paragraphs[0]
    p.text = q_en
    p.font.size = Pt(10)
    p.font.color.rgb = ACCENT
    p.font.bold = True
    add_para(qtf, answer, 9, DARK, space_before=Pt(3))

sn(sl, 12)

# ── Save ─────────────────────────────────────────────────────────────

out_path = os.path.join(BASE, "KSAS_Spring2026_Presentation.pptx")
prs.save(out_path)
print(f"Saved: {out_path}")
