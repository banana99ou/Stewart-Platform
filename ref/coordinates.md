### origin @ top of bottom plate.

for Servo1:

BottomAchor1(center of servo axle) -50, 100, 10\
hornLength 35 (perpendicular to servo axis)\
hornOffset(thickness of servo horn) 11.3 (palallel to servo axis)\
ball link dia(thickness of ball link attached to servo horn. it connects to Linkage Rod) 3.3 \
so when servo1 @0 degrees, servo horn ball link center @(-85, 108+3.3/2, 10)
Linkage 162.21128 (ball link center to center) \
UpperAchor1(center of ball link connected to Upper plate) -7.5, 108+3.3/2, 152.5

there is 6 servos, 6 servo horn, 6 Linkage, 12 ball links.
these parts are layed out in triangular pattern and makes stewart platform.
servo 3 5 is 120 degree rotation about Z axis of servo1. and servo 2 4 6 is 120 degree rotation about Z axis of x axis mirror of servo1.

origin is at the center of stewart platform. and is placed on top of bottom plate.
servos are connected to bottom plate and numberd from top left to counter clockwise.
servos connects to linkages via ball link. and ball links and linkages connects Upper plate.