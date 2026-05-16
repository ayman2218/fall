"""L298N motor driver helper for Raspberry Pi GPIO (BCM).

This module provides a simple differential-drive interface:
- set_speeds(left, right) where left/right are in [-1.0, +1.0]

Assumptions:
- ENA and ENB are connected to PWM-capable GPIOs
- IN1/IN2 control left motor direction
- IN3/IN4 control right motor direction

Requires: RPi.GPIO
"""

from __future__ import annotations

import time

try:
    import RPi.GPIO as GPIO
except Exception as e:
    raise RuntimeError(
        "RPi.GPIO not available. Run this on Raspberry Pi OS with RPi.GPIO installed."
    ) from e


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class L298NMotorDriver:
    def __init__(
        self,
        *,
        left_en_pwm: int,
        left_in1: int,
        left_in2: int,
        right_en_pwm: int,
        right_in3: int,
        right_in4: int,
        pwm_frequency_hz: int = 1000,
        invert_left: bool = False,
        invert_right: bool = False,
    ):
        self.left_en_pwm = left_en_pwm
        self.left_in1 = left_in1
        self.left_in2 = left_in2
        self.right_en_pwm = right_en_pwm
        self.right_in3 = right_in3
        self.right_in4 = right_in4
        self.pwm_frequency_hz = pwm_frequency_hz
        self.invert_left = invert_left
        self.invert_right = invert_right

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in (
            self.left_in1,
            self.left_in2,
            self.right_in3,
            self.right_in4,
            self.left_en_pwm,
            self.right_en_pwm,
        ):
            GPIO.setup(pin, GPIO.OUT)

        # Start with motors off
        GPIO.output(self.left_in1, GPIO.LOW)
        GPIO.output(self.left_in2, GPIO.LOW)
        GPIO.output(self.right_in3, GPIO.LOW)
        GPIO.output(self.right_in4, GPIO.LOW)

        self._pwm_left = GPIO.PWM(self.left_en_pwm, self.pwm_frequency_hz)
        self._pwm_right = GPIO.PWM(self.right_en_pwm, self.pwm_frequency_hz)
        self._pwm_left.start(0)
        self._pwm_right.start(0)

        # For gentle changes, keep last values (optional smoothing done outside)
        self._last_left = 0.0
        self._last_right = 0.0

    def stop(self):
        self.set_speeds(0.0, 0.0)

    def set_speeds(self, left: float, right: float):
        """Set motor speeds.

        left/right in [-1.0, +1.0]
          + => forward
          - => backward
        """
        left = _clamp(left, -1.0, 1.0)
        right = _clamp(right, -1.0, 1.0)

        if self.invert_left:
            left = -left
        if self.invert_right:
            right = -right

        # Direction pins
        self._set_dir(self.left_in1, self.left_in2, left)
        self._set_dir(self.right_in3, self.right_in4, right)

        # PWM duty cycle: 0..100
        duty_left = abs(left) * 100.0
        duty_right = abs(right) * 100.0
        self._pwm_left.ChangeDutyCycle(duty_left)
        self._pwm_right.ChangeDutyCycle(duty_right)

        self._last_left = left
        self._last_right = right

    @staticmethod
    def _set_dir(in_a: int, in_b: int, value: float):
        if value > 0:
            GPIO.output(in_a, GPIO.HIGH)
            GPIO.output(in_b, GPIO.LOW)
        elif value < 0:
            GPIO.output(in_a, GPIO.LOW)
            GPIO.output(in_b, GPIO.HIGH)
        else:
            GPIO.output(in_a, GPIO.LOW)
            GPIO.output(in_b, GPIO.LOW)

    def close(self):
        try:
            self.stop()
            time.sleep(0.05)
            self._pwm_left.stop()
            self._pwm_right.stop()
        finally:
            GPIO.cleanup()
