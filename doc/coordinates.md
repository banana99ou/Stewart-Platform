Right-handed cartesian coordinate (Z up)
X+ is right, Y+ is back/away, Z+ is up
origin @ top of bottom plate.

for Servo1:

BottomAchor1(center of servo axle) -50, 100, 10\
hornLength 35 (perpendicular to servo axis)\
hornOffset(thickness of servo horn) 11.3 (palallel to servo axis)\
ball link dia(thickness of ball link attached to servo horn. it connects to Linkage Rod) 3.3 \
so when servo1 @0 degrees, servo horn ball link center @(-97.5, 108+3.3/2, 10)
Linkage 127.0 (ball link center to center) \
UpperAchor1(center of ball link connected to Upper plate) -7.5, 108+3.3/2, 99.60469

there is 6 servos, 6 servo horn, 6 Linkage, 12 ball links.
these parts are layed out in triangular pattern and makes stewart platform.
servo 3 5 is 120 degree rotation about Z axis of servo1. and servo 2 4 6 is 120 degree rotation about Z axis of x axis mirror of servo1.

origin is at the center of stewart platform. and is placed on top of bottom plate.
servos are connected to bottom plate and numberd from top left to counter clockwise.
servos connects to linkages via ball link. and ball links and linkages connects Upper plate.


## Actuation range (from IK sweeps, assuming ±90° servo limits)

Single-DOF sweeps with the other DOFs held at 0 give the following approximate
reachable ranges where all 6 legs have a valid IK solution within ±90°:

- translation X (`dx`): about 
- translation Y (`dy`): about 
- translation Z (`dz`): about 
- pitch about X (`pitch`): about 
- roll about Y (`roll`): about 
- yaw about Z (`yaw`): about 

These are *per-axis* ranges; simultaneous motions in multiple DOFs will have a
smaller feasible region due to intersection of constraints.