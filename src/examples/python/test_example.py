#!/usr/bin/env python3
"""Test the example nanobind module."""

import example


def test_add():
    """Test basic integer addition."""
    assert example.add(2, 3) == 5
    assert example.add(-1, 1) == 0
    assert example.add(0, 0) == 0
    print("  add: pass")


def test_greet():
    """Test string operations."""
    result = example.greet("World")
    assert "World" in result
    assert "aleph" in result
    print("  greet: pass")


def test_dot_product():
    """Test vector dot product."""
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    result = example.dot_product(a, b)
    expected = 1 * 4 + 2 * 5 + 3 * 6  # = 32
    assert abs(result - expected) < 1e-10
    print("  dot_product: pass")


def test_norm():
    """Test vector norm."""
    v = [3.0, 4.0]
    result = example.norm(v)
    expected = 5.0  # 3-4-5 triangle
    assert abs(result - expected) < 1e-10
    print("  norm: pass")


def test_counter():
    """Test Counter class."""
    c = example.Counter(10)
    assert c.get() == 10

    c.increment()
    assert c.get() == 11

    c.decrement()
    assert c.get() == 10

    c.add(5)
    assert c.get() == 15

    c.reset()
    assert c.get() == 0

    # Test repr
    c = example.Counter(42)
    assert "42" in repr(c)
    print("  Counter: pass")


def main():
    """Run all tests."""
    print("Python calling C++ via nanobind:")
    test_add()
    test_greet()
    test_dot_product()
    test_norm()
    test_counter()
    print("all tests passed")


if __name__ == "__main__":
    main()
