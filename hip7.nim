# 2017/12/13
# Test HIP addresses
# William Whitacre

import sequtils, future, algorithm, math

import arraymancer
import arraymancer_vision

let
  sqrt3: float = 1.732050807568877293527446341505872366942805253810380628055
  TEST_EPSILON: float = 1e-10
  #sqrt7: float = 2.645751311064590590501615753639260425710259183082450180368
  #atan_sqrt3_2: float =
  #    0.713724378944765630818123705741566557999822743277044291020

type
  CodedRadix7Addr*[T] = distinct T

proc radix7*[T:not(seq|array)](n: T): CodedRadix7Addr[T] =
  var
    x = abs(n)
    r = 0
    c = 1

  while x > 0:
    r += (x %% 10) * c
    c *= 7
    x = x /% 10

  return CodedRadix7Addr(r)

proc radix7*[T:seq|array](n: T): auto =
  result = map(n, radix7)

proc radix7*[T](n: Tensor[T]): Tensor[CodedRadix7Addr[T]] =
  result = map(n, radix7)

let radix7_add_table =
  radix7(@[@[0,  1,  2,  3,  4,  5,  6],
          @[1, 63, 15,  2,  0,  6, 64],
          @[2, 15, 14, 26,  3,  0,  1],
          @[3,  2, 26, 25, 31,  4,  0],
          @[4,  0,  3, 31, 36, 42,  5],
          @[5,  6,  0,  4, 42, 41, 53],
          @[6, 64,  1,  0,  5, 53, 52]])

let radix7_mul_table =
  radix7(@[@[0, 0, 0, 0, 0, 0, 0],
          @[0, 1, 2, 3, 4, 5, 6],
          @[0, 2, 3, 4, 5, 6, 1],
          @[0, 3, 4, 5, 6, 1, 2],
          @[0, 4, 5, 6, 1, 2, 3],
          @[0, 5, 6, 1, 2, 3, 4],
          @[0, 6, 1, 2, 3, 4, 5]])

iterator radix7_digits_iter*[T](x: CodedRadix7Addr[T]): T =
  var y = x
  if y == 0:
    yield 0
  else:
    while y > 0:
      yield T(int(y) %% 7)
      y = T(int(y) /% 7)

proc `$`*[T](x: CodedRadix7Addr[T]): string =
  result = "HIP{"
  var u = ""
  for d in radix7_digits_iter(T(x)):
    u = $d & u
  result &= u & "}"

proc add_radix7_digits[T](a, b: T): T =
  result = T(radix7_add_table[a][b])

proc mul_radix7_digits[T](a, b: T): T =
  result = T(radix7_mul_table[a][b])

proc add_radix7_nums[T](a, b: T): T =
  var
    A: int = a
    B: int = b
    c: int = 0
    v: int = 0
    r: int = 0
    o: int = 1

  if a == 0:
    return b
  elif b == 0:
    return a

  while c > 0 or A > 0 or B > 0:
    v = add_radix7_nums(add_radix7_digits(A %% 7, B %% 7), c)
    r += (v %% 7) * o
    c = v /% 7
    A = A /% 7
    B = B /% 7
    o *= 7

  result = r

proc mul_radix7_inner[T](a, b: T): T =
  var
    r: int = 0
    o: int = 1

  for db in radix7_digits_iter(b):
    r += o * mul_radix7_digits(db, a)
    o *= 7

  result = r

proc mul_radix7_nums[T](a, b: T): T =
  var
    o: int = 1
    r: int = 0
    u: int = 0

  for da in radix7_digits_iter(a):
    u = o * mul_radix7_inner(da, b)
    r = add_radix7_nums(u, r)
    o *= 7

  result = r

proc `+`*[T](a, b: CodedRadix7Addr[T]): CodedRadix7Addr[T] =
  result = CodedRadix7Addr(add_radix7_nums(T(a), T(b)))

proc `*`*[T](a, b: CodedRadix7Addr[T]): CodedRadix7Addr[T] =
  result = CodedRadix7Addr(mul_radix7_nums(T(a), T(b)))

# Bh coordinates
#
# skewed axes at angle 2pi/3
proc as_skew_tensor*[T](adr: CodedRadix7Addr[T]): auto =
  let
    Eye = to_tensor([[1'i64, 0'i64],[0'i64, 1'i64]])
    M = to_tensor([[3'i64, -2'i64], [2'i64, 1'i64]])
    unitt = to_tensor([1'i64, 0'i64])
    rottab = [
      to_tensor([[ 0'i64,  0'i64], [ 0'i64,  0'i64]]), # 0
      to_tensor([[ 1'i64,  0'i64], [ 0'i64,  1'i64]]), # 1
      to_tensor([[ 1'i64, -1'i64], [ 1'i64,  0'i64]]), # 2
      to_tensor([[ 0'i64, -1'i64], [ 1'i64, -1'i64]]), # 3
      to_tensor([[-1'i64,  0'i64], [ 0'i64, -1'i64]]), # 4
      to_tensor([[-1'i64,  1'i64], [-1'i64,  0'i64]]), # 5
      to_tensor([[ 0'i64,  1'i64], [-1'i64,  1'i64]])  # 6
      ]

  var
    m = clone(Eye)
    r = zeros_like(unitt)

  for da in radix7_digits_iter(T(adr)):
    if da != 0:
      r += rottab[da] * m * unitt
    m = m * M

  result = r

proc as_skew*[T](adr: CodedRadix7Addr[T]): auto =
  let r = as_skew_tensor(adr)
  result = (bx: r[0], by: r[1])

proc as_real_tensor*[T](adr: CodedRadix7Addr[T]): auto =
  let sk = to_tensor([[2.0, -1.0], [0.0, sqrt3]]).as_type(float) * 0.5
  result = sk * as_skew_tensor(adr).as_type(float)

proc as_real_direct*[T](adr: CodedRadix7Addr[T]): auto =
  var
    xc: float = 1.0
    yc: float = 0.0
  
    x: float = 0.0
    y: float = 0.0

    t1: float
    t2: float

  for da in radix7_digits_iter(T(adr)):
    # first has least influence
    if da == 1:
      x = x + xc
      y = y + yc
    elif da == 2:
      x = x + (xc * 0.5) - ((sqrt3 * yc) * 0.5)
      y = y + ((sqrt3*xc) * 0.5) + (yc * 0.5)
    elif da == 3:
      x = x + (xc * -0.5) - ((sqrt3 * yc) * 0.5)
      y = y + ((sqrt3 * xc) * 0.5) - (yc * 0.5)
    elif da == 4:
      x = x - xc
      y = y - yc
    elif da == 5:
      x = x + (xc * -0.5) + ((sqrt3 * yc) * 0.5)
      y = y + ((sqrt3 * xc) * -0.5) - (yc * 0.5)
    elif da == 6:
      x = x + (xc * 0.5) + ((sqrt3 * yc) * 0.5)
      y = y + ((sqrt3 * xc) * -0.5) + (yc * 0.5)
    elif da != 0:
      raise newException(ValueError, "HIP7 digit out of range.")
    
    t1 = 2.0*xc - sqrt3*yc
    t2 = sqrt3*xc + 2.0*yc
    xc = t1
    yc = t2

  result = (x: x, y: y)

proc as_real*[T](adr: CodedRadix7Addr[T]): auto =
  var r = as_real_tensor(adr)
  result = (x: r[0], y: r[1])

proc as_polar_tensor*[T](adr: CodedRadix7Addr[T]): auto =
  let r2 = as_real_tensor(adr)
  result = to_tensor([sqrt(r2[0]*r2[0] + r2[1]*r2[1]), arctan2(r2[1], r2[0])])

proc as_polar*[T](adr: CodedRadix7Addr[T]): auto =
  let
    pol = as_polar_tensor(adr)
  result = (r: pol[0], t: pol[1])

proc hip_clamp_angle*[T](angle: T): auto =
  ((angle + PI) mod TAU) - PI

# Unit tests
when isMainModule:
  import unittest

  let
    atan_sqrt3_2 = arctan(sqrt(3.0)/2.0)
    sqrt7 = sqrt(7.0)

  suite "Numerical stability":
    test "Ascending locus of aggregate centers":
      var v1 = as_polar(CodedRadix7Addr(1))
      for i in 1||7:
        var v2 = as_polar(CodedRadix7Addr(7^i))
        var
          r1 = v1[0]
          t1 = v1[1]
          r2 = v2[0]
          t2 = v2[1]

        check: abs(hip_clamp_angle(t2 - t1)) - atan_sqrt3_2 < TEST_EPSILON
        check: abs(r1 * sqrt7 - r2) < TEST_EPSILON

        v1 = v2

    test "Skew coordinates accuracy":
      for i in 0||(7^4):
        let
          directcoordstup = as_real_direct(CodedRadix7Addr(i))
          fromskewcoords = as_real_tensor(CodedRadix7Addr(i))
          directcoords = to_tensor([directcoordstup[0], directcoordstup[1]])
        
          err = max(abs(directcoords - fromskewcoords))
        
        check: err < TEST_EPSILON

  #for i in 0..(7^2):
  #  echo CodedRadix7Addr(i)
  #  echo "SKEW COORDS : ", as_skew(CodedRadix7Addr(i))
  #  echo "CARTESIAN COORDS (DIRECT) : ", as_real_direct(CodedRadix7Addr(i))
  #  echo "CARTESIAN COORDS (SKEWC)  : ", as_real(CodedRadix7Addr(i))
  #  echo "POLAR COORDS : ", as_polar(CodedRadix7Addr(i))
  #
  #for i in 0..(7^2):
  #  echo radix7(10), " + ", CodedRadix7Addr(i), " = ", radix7(10) + CodedRadix7Addr(i)
  #  echo "SKEW COORDS : ", as_skew(radix7(10) + CodedRadix7Addr(i))
  #  echo "CARTESIAN COORDS (DIRECT) : ", as_real_direct(radix7(10) + CodedRadix7Addr(i))
  #  echo "CARTESIAN COORDS (SKEWC)  : ", as_real(radix7(10) + CodedRadix7Addr(i))
  #  echo "POLAR COORDS : ", as_polar(radix7(10) + CodedRadix7Addr(i))
  #
  #for i in 0..(7^2):
  #  echo radix7(10), " * ", CodedRadix7Addr(i), " = ", radix7(10) * CodedRadix7Addr(i)
  #  echo "SKEW COORDS : ", as_skew(radix7(10) * CodedRadix7Addr(i))
  #  echo "CARTESIAN COORDS (DIRECT) : ", as_real_direct(radix7(10) * CodedRadix7Addr(i))
  #  echo "CARTESIAN COORDS (SKEWC)  : ", as_real(radix7(10) * CodedRadix7Addr(i))
  #  echo "POLAR COORDS : ", as_polar(radix7(10) * CodedRadix7Addr(i))

  #echo(to_tensor([[3'i32, -2'i32], [2'i32, 1'i32]]))
  #echo(to_tensor([1'i32, 0'i32]))

  #echo(to_tensor([[3'i32, -2'i32], [2'i32, 1'i32]]) * to_tensor([1'i32, 0'i32]))
