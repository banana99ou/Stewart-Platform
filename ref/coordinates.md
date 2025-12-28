In left handed cartesian coordinate
X+ is right y+ is back/away z+ is up
origin @ top of bottom plate.

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


## Actuation range (from IK sweeps, assuming ±90° servo limits)

Single-DOF sweeps with the other DOFs held at 0 give the following approximate
reachable ranges where all 6 legs have a valid IK solution within ±90°:

- translation X (`dx`): about **−78 mm to +78 mm**
- translation Y (`dy`): about **−82 mm to +82 mm**
- translation Z (`dz`): about **−22 mm to +49 mm**
- roll about X (`roll`): about **−11° to +23°**
- pitch about Y (`pitch`): about **−13° to +13°**
- yaw about Z (`yaw`): about **−42° to +42°**

These are *per-axis* ranges; simultaneous motions in multiple DOFs will have a
smaller feasible region due to intersection of constraints.