# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
# TODO: Replace these demo tests with tests for your own code.
from smcjax import add, hello, multiply, subtract


def test_hello_with_custom_name_returns_greeting():
    """Test that hello returns greeting when given a custom name."""
    assert hello('World!') == 'Hello World!'


def test_hello_default_name_returns_hello_world():
    """Test that hello returns 'Hello world' when called with no args."""
    assert hello() == 'Hello world'


def test_hello_with_empty_string_returns_hello():
    """Test that hello returns 'Hello ' when given an empty string."""
    assert hello('') == 'Hello '


def test_add_two_positive_integers_returns_sum():
    """Test that add returns the sum of two positive integers."""
    assert add(1, 2) == 3


def test_add_with_zero_returns_same_number():
    """Test that add returns the same number when adding zero."""
    assert add(5, 0) == 5


def test_add_negative_numbers_returns_sum():
    """Test that add returns the correct sum of two negative numbers."""
    assert add(-3, -7) == -10


def test_subtract_two_positive_integers_returns_difference():
    """Test that subtract returns the difference of two positive ints."""
    assert subtract(5, 3) == 2


def test_subtract_equal_integers_returns_zero():
    """Test that subtract returns zero when both operands are equal."""
    assert subtract(3, 3) == 0


def test_subtract_larger_subtrahend_returns_negative():
    """Test that subtract returns negative when subtrahend is larger."""
    assert subtract(1, 5) == -4


def test_multiply_two_positive_integers_returns_product():
    """Test that multiply returns the product of two positive integers."""
    assert multiply(3, 4) == 12


def test_multiply_by_zero_returns_zero():
    """Test that multiply returns zero when one operand is zero."""
    assert multiply(5, 0) == 0


def test_multiply_negative_integer_returns_correct_product():
    """Test that multiply handles negative integers correctly."""
    assert multiply(-3, 4) == -12
