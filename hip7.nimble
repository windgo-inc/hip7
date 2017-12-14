# Package

version       = "0.1.0"
author        = "William Whitacre"
description   = "Hexagonal image processing using arraymancer and arraymancer_vision."
license       = "MIT"

# Dependencies

requires "nim >= 0.17.2"
requires "arraymancer >= 0.2.0"
requires "arraymancer_vision >= 0.0.3"

skipDirs = @["tests"]

#task test_stability, "Testing numerical stability":
#  test "test_stability"
