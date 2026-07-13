# RND STEP URDF 기본 자세 및 관절 좌표계

이 문서 하나만으로 RND STEP 로봇의 기본 자세와 관절 좌표 부호를 동일하게 구현할 수 있도록 필요한
수치와 변환 규칙을 모두 정의한다.

## 1. 로봇 관절 구성

가동 관절은 다리의 revolute joint 12개다.

```text
R_Leg_hip_yaw
R_Leg_hip_roll
R_Leg_hip_pitch
R_Leg_knee
R_Leg_ankle_pitch
R_Leg_ankle_roll
L_Leg_hip_yaw
L_Leg_hip_roll
L_Leg_hip_pitch
L_Leg_knee
L_Leg_ankle_pitch
L_Leg_ankle_roll
```

Waist, 양팔, 손, 목은 현재 fixed joint다. 다만 동일한 팔 자세를 revolute-joint 모델에서도 재현할 수
있도록 4.3절에 fixed rotation을 `zero alignment * joint rotation`으로 분해한 등가 팔 관절값을 정의한다.

관절 위치 단위는 radian, 관절 속도 단위는 rad/s다.

## 2. Base 좌표계

`base_link` 좌표계를 다음과 같이 사용한다.

```text
+X_B : 로봇의 왼쪽
-X_B : 로봇의 오른쪽
-Y_B : 로봇의 전방
+Y_B : 로봇의 후방
+Z_B : 위쪽
-Z_B : 아래쪽
```

따라서 이 로봇의 전방은 `+X_B`가 아니라 **`-Y_B`**다.

## 3. Revolute joint 회전 규약

각 revolute joint의 local rotation axis는 다음과 같다.

```text
axis_local = (0, 0, 1)
```

이 축은 base frame의 Z축이 아니라 joint origin으로 회전된 **joint local frame의 Z축**이다. 모든 joint를
base Z축 회전으로 해석하면 안 된다.

parent frame에서 child frame으로의 회전은 다음 순서로 계산한다.

```text
R_origin = Rz(yaw) * Ry(pitch) * Rx(roll)

R_parent_child(q) = R_origin * Rot(axis_local, q)
```

- `q = 0`: joint origin의 고정 xyz/rpy만 적용된 상태
- `q > 0`: joint axis를 향해 오른손 엄지를 놓았을 때 손가락이 감기는 방향
- joint가 회전하면 해당 child와 모든 descendant link가 함께 회전

현재 자세에서 joint axis를 base frame으로 표현하려면 다음 식을 사용한다.

```text
axis_B(q) = R_B_parent(q_upstream) * R_origin * axis_local
```

`axis_B`는 upstream 관절각에 따라 변하므로 동역학 계산에서 매 시점 forward kinematics로 갱신해야 한다.

## 4. 표준 기본 자세

### 4.1 Base 상태

```text
base position    = (0.0, 0.0, 0.44) m
base quaternion  = (w, x, y, z) = (1, 0, 0, 0)
joint velocity   = 0 rad/s for every movable joint
```

### 4.2 관절 기본각

| Joint name | 기본각 (rad) | 기본각 (deg) | Motor ID | Encoder direction | 기본각 대응 raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| `R_Leg_hip_yaw` | 0.000000 | 0.000 | 10 | +1 | 2048 |
| `R_Leg_hip_roll` | +0.100000 | +5.730 | 15 | -1 | 1983 |
| `R_Leg_hip_pitch` | -0.630000 | -36.096 | 13 | +1 | 1637 |
| `R_Leg_knee` | +0.150000 | +8.594 | 17 | -1 | 1950 |
| `R_Leg_ankle_pitch` | -0.910000 | -52.139 | 19 | -1 | 2641 |
| `R_Leg_ankle_roll` | -0.100000 | -5.730 | 21 | +1 | 1983 |
| `L_Leg_hip_yaw` | 0.000000 | 0.000 | 12 | +1 | 2048 |
| `L_Leg_hip_roll` | -0.100000 | -5.730 | 16 | -1 | 2113 |
| `L_Leg_hip_pitch` | +0.630000 | +36.096 | 14 | +1 | 2459 |
| `L_Leg_knee` | -0.150000 | -8.594 | 18 | -1 | 2146 |
| `L_Leg_ankle_pitch` | +0.910000 | +52.139 | 20 | -1 | 1455 |
| `L_Leg_ankle_roll` | +0.100000 | +5.730 | 22 | +1 | 2113 |

기본 자세의 좌우 관계는 다음과 같다.

```text
q_L_hip_roll     = -q_R_hip_roll
q_L_hip_pitch    = -q_R_hip_pitch
q_L_knee         = -q_R_knee
q_L_ankle_pitch  = -q_R_ankle_pitch
q_L_ankle_roll   = -q_R_ankle_roll
q_L_hip_yaw      =  q_R_hip_yaw = 0
```

오른쪽과 왼쪽 pitch 계열의 회전축이 서로 반대이므로, 동일한 물리적 sagittal 자세를 만들 때 관절각 부호도
서로 반대가 된다.

### 4.3 팔의 ㄴ자 전방 자세

팔을 가동 관절로 구성하는 모델에서는 다음 6개 등가 관절값을 사용한다.

| Joint name | 기본각 (rad) | 기본각 (deg) | 자세에서의 역할 |
| --- | ---: | ---: | --- |
| `R_Arm_shoulder_yaw` | 0.000000 | 0.000 | 오른쪽 shoulder 기준 정렬 유지 |
| `R_Arm_pitch` | +1.300000 | +74.485 | 오른쪽 upper arm을 ㄴ자 자세의 shoulder 위치로 회전 |
| `R_Arm_elbow` | +1.570000 | +89.954 | 오른쪽 팔꿈치를 약 90도로 굽힘 |
| `L_Arm_shoulder_yaw` | 0.000000 | 0.000 | 왼쪽 shoulder 기준 정렬 유지 |
| `L_Arm_pitch` | -1.300000 | -74.485 | 왼쪽 upper arm을 오른쪽과 대칭으로 회전 |
| `L_Arm_elbow` | -1.570000 | -89.954 | 왼쪽 팔꿈치를 오른쪽과 대칭으로 약 90도 굽힘 |

이 값만으로 자세를 재현하려면 각 revolute joint의 `q=0` frame도 다음과 같이 정의해야 한다. local axis는
모두 `(0, 0, 1)`이며, 아래 `zero-alignment RPY`를 먼저 적용한 뒤 해당 `q_default`만큼 local Z축 회전을
적용한다.

```text
R_joint(q) = Rz(yaw0) * Ry(pitch0) * Rx(roll0) * Rz(q)
```

| Joint name | zero-alignment RPY (rad) | axis_local | q_default (rad) |
| --- | --- | --- | ---: |
| `R_Arm_shoulder_yaw` | `(0, -1.5708, 0)` | `(0, 0, 1)` | 0.000000 |
| `R_Arm_pitch` | `(-1.5708, 0, -3.1416)` | `(0, 0, 1)` | +1.300000 |
| `R_Arm_elbow` | `(+3.1416, +1.5708, 0)` | `(0, 0, 1)` | +1.570000 |
| `L_Arm_shoulder_yaw` | `(0, +1.5708, 0)` | `(0, 0, 1)` | 0.000000 |
| `L_Arm_pitch` | `(-1.5708, 0, -3.1416)` | `(0, 0, 1)` | -1.300000 |
| `L_Arm_elbow` | `(+3.1416, -1.5708, 0)` | `(0, 0, 1)` | -1.570000 |

이 자세에서 양쪽 forearm은 팔꿈치에서 손 방향으로 거의 `-Y_B`를 향한다. 즉, body 전방으로 약 79 mm
뻗어 있으므로 여기서 말하는 기본 팔 자세는 **팔꿈치를 약 90도로 굽히고 forearm을 전방으로 향하게 한
ㄴ자 자세**다.

팔의 좌우 관계는 다음과 같다.

```text
q_L_shoulder_yaw =  q_R_shoulder_yaw = 0
q_L_arm_pitch    = -q_R_arm_pitch
q_L_arm_elbow    = -q_R_arm_elbow
```

## 5. 기본 자세에서의 양의 회전 방향

다음 `axis_B`는 4절의 기본 자세를 적용했을 때 각 관절의 양의 회전축을 base frame으로 표현한 값이다.

| Joint name | 기본 자세의 `axis_B` | `q`가 증가할 때의 회전 |
| --- | --- | --- |
| `R_Leg_hip_yaw` | `(0, 0, -1)` | 위에서 내려다보면 시계 방향 yaw |
| `R_Leg_hip_roll` | `(0, +1, 0)` | 오른쪽 다리를 `-X_B`, 즉 로봇 바깥쪽으로 이동시키는 방향 |
| `R_Leg_hip_pitch` | `(+0.9950, 0, -0.0998)` | 주로 `+X_B`축 회전. 오른쪽 thigh가 후방 `+Y_B` 쪽으로 회전 |
| `R_Leg_knee` | `(+0.9950, 0, -0.0998)` | 오른쪽 calf와 하위 링크가 해당 축의 오른손 방향으로 회전 |
| `R_Leg_ankle_pitch` | `(+0.9950, 0, -0.0998)` | 오른쪽 ankle과 foot이 해당 축의 오른손 방향으로 회전 |
| `R_Leg_ankle_roll` | `(+0.0015, +0.9999, +0.0154)` | 거의 `+Y_B`축을 중심으로 오른쪽 foot이 회전 |
| `L_Leg_hip_yaw` | `(0, 0, -1)` | 위에서 내려다보면 시계 방향 yaw |
| `L_Leg_hip_roll` | `(0, +1, 0)` | 왼쪽 다리를 `-X_B`, 즉 몸 안쪽으로 이동. 바깥쪽은 `q < 0` |
| `L_Leg_hip_pitch` | `(-0.9950, 0, -0.0998)` | 주로 `-X_B`축 회전. 왼쪽 thigh가 전방 `-Y_B` 쪽으로 회전 |
| `L_Leg_knee` | `(-0.9950, 0, -0.0998)` | 왼쪽 calf와 하위 링크가 해당 축의 오른손 방향으로 회전 |
| `L_Leg_ankle_pitch` | `(-0.9950, 0, -0.0998)` | 왼쪽 ankle과 foot이 해당 축의 오른손 방향으로 회전 |
| `L_Leg_ankle_roll` | `(-0.0016, +0.9999, +0.0160)` | 거의 `+Y_B`축을 중심으로 왼쪽 foot이 회전 |

핵심 부호 규칙은 다음과 같다.

```text
hip yaw:
  좌우 모두 q 증가 = -Z_B축 양의 회전

hip roll:
  좌우 모두 q 증가 = +Y_B축 양의 회전
  오른쪽 다리 바깥쪽 = q_R > 0
  왼쪽 다리 바깥쪽   = q_L < 0

pitch / knee / ankle pitch:
  오른쪽 양의 축은 주로 +X_B
  왼쪽 양의 축은 주로 -X_B
  좌우 대칭 동작에는 반대 부호의 q를 사용

ankle roll:
  기본 자세에서 좌우 축은 모두 거의 +Y_B
  좌우 대칭 foot 자세에는 반대 부호의 q를 사용
```

팔을 4.3절의 revolute-joint 분해로 표현했을 때 기본 자세의 양의 축은 다음과 같다.

| Joint name | 기본 자세의 `axis_B` |
| --- | --- |
| `R_Arm_shoulder_yaw` | `(-1, 0, 0)` |
| `R_Arm_pitch` | `(0, -1, 0)` |
| `R_Arm_elbow` | `(-0.9636, 0, +0.2675)` |
| `L_Arm_shoulder_yaw` | `(+1, 0, 0)` |
| `L_Arm_pitch` | `(0, -1, 0)` |
| `L_Arm_elbow` | `(+0.9636, 0, +0.2675)` |

팔 관절도 각 `axis_B`의 오른손 법칙을 따른다. 팔꿈치의 오른쪽 `q > 0`, 왼쪽 `q < 0`가 서로 대칭인
약 90도 굽힘을 만든다.

## 6. Encoder raw와 관절각 변환

### 6.1 공통 상수

```text
ticks_per_revolution = 4096
zero_raw             = 2048 for every joint
electrical raw range = 0 ... 4095

rad_per_tick = 2*pi/4096
             = 0.0015339808 rad

deg_per_tick = 360/4096
             = 0.087890625 deg
```

### 6.2 변환식

```text
q_rad = direction * (raw - zero_raw) * (2*pi / 4096)

raw_goal = round(
    zero_raw + direction * q_goal_rad / (2*pi / 4096)
)
```

- `direction = +1`: raw가 증가하면 `q`가 증가
- `direction = -1`: raw가 감소하면 `q`가 증가
- `direction`은 encoder 좌표와 관절 좌표 사이의 부호이며 토크 부호가 아니다.

### 6.3 관절별 encoder 부호

| Joint name | ID | zero_raw | direction | raw가 1 증가할 때 `q` 변화 |
| --- | ---: | ---: | ---: | --- |
| `R_Leg_hip_yaw` | 10 | 2048 | +1 | `+0.0015339808 rad` |
| `R_Leg_hip_roll` | 15 | 2048 | -1 | `-0.0015339808 rad` |
| `R_Leg_hip_pitch` | 13 | 2048 | +1 | `+0.0015339808 rad` |
| `R_Leg_knee` | 17 | 2048 | -1 | `-0.0015339808 rad` |
| `R_Leg_ankle_pitch` | 19 | 2048 | -1 | `-0.0015339808 rad` |
| `R_Leg_ankle_roll` | 21 | 2048 | +1 | `+0.0015339808 rad` |
| `L_Leg_hip_yaw` | 12 | 2048 | +1 | `+0.0015339808 rad` |
| `L_Leg_hip_roll` | 16 | 2048 | -1 | `-0.0015339808 rad` |
| `L_Leg_hip_pitch` | 14 | 2048 | +1 | `+0.0015339808 rad` |
| `L_Leg_knee` | 18 | 2048 | -1 | `-0.0015339808 rad` |
| `L_Leg_ankle_pitch` | 20 | 2048 | -1 | `-0.0015339808 rad` |
| `L_Leg_ankle_roll` | 22 | 2048 | +1 | `+0.0015339808 rad` |

예시:

```text
R_Leg_knee:
  q_default = +0.15 rad
  direction = -1
  raw_default = round(2048 - 0.15/(2*pi/4096)) = 1950

L_Leg_knee:
  q_default = -0.15 rad
  direction = -1
  raw_default = round(2048 + 0.15/(2*pi/4096)) = 2146
```

`0 ... 4095`는 encoder의 전기적 범위다. 로봇 링크가 충돌하지 않는 기계적 안전 범위를 의미하지 않는다.

## 7. 기본 자세를 동일하게 적용하는 순서

1. base frame을 2절과 동일하게 정의한다.
2. 12개 joint 이름과 parent-child 연결을 동일하게 유지한다.
3. joint origin 회전 후 local `+Z`축에 대해 `q`를 적용한다.
4. 상태를 encoder raw로 받는 경우 6절의 변환식으로 radian 좌표로 바꾼다.
5. 4.2절의 radian 값을 목표 기본 자세로 사용한다.
6. 실물에서는 현재 관절각에서 기본 자세까지 연속 trajectory로 이동한다.
7. 팔이 fixed인 모델에서는 팔에 command를 보내지 않는다. 팔을 revolute joint로 구성한 모델에서는
   4.3절의 zero alignment와 기본각을 함께 적용한다.
8. Waist, hand, neck이 fixed라면 해당 joint에는 목표 관절각을 보내지 않는다.

현재 자세 `q_start`에서 기본 자세 `q_default`로 이동할 때 사용할 수 있는 연속 보간식은 다음과 같다.

```text
tau = clamp(t / duration, 0, 1)
s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5

q_cmd(t) = q_start + s(tau) * (q_default - q_start)
```

기본 raw 값을 모든 모터에 한 번에 즉시 쓰지 않는다. 보간 시간과 기계적 joint limit는 실제 링크 간섭과
허용 속도에 맞춰 별도로 정한다.

## 8. 최종 확인 항목

```text
[ ] body 전방을 -Y_B로 사용했는가?
[ ] raw=2048일 때 모든 가동 joint가 q=0 rad가 되는가?
[ ] direction=-1 joint에서 raw 증가가 q 감소로 변환되는가?
[ ] 좌우 pitch 계열에 반대 부호를 적용했는가?
[ ] joint local axis (0,0,1)을 base Z축으로 오해하지 않았는가?
[ ] 동역학 계산에서 axis_B를 현재 자세에 따라 갱신하는가?
[ ] 팔이 가동형이면 pitch와 elbow에 좌우 반대 부호를 적용했는가?
[ ] 팔의 zero alignment와 q_default를 별개의 회전으로 적용했는가?
[ ] 고정형 upper-body joint에 command를 보내지 않는가?
[ ] encoder 전기적 범위와 기계적 안전 범위를 구분했는가?
```
