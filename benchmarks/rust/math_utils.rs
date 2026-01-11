//! Math Utilities - Benchmark Rust Program
//! A collection of mathematical functions for test generation evaluation

/// Safe addition with overflow checking
pub fn safe_add(a: i32, b: i32) -> Option<i32> {
    a.checked_add(b)
}

/// Safe multiplication with overflow checking
pub fn safe_multiply(a: i32, b: i32) -> Option<i32> {
    a.checked_mul(b)
}

/// Safe division returning None for division by zero
pub fn safe_divide(numerator: i32, denominator: i32) -> Option<i32> {
    if denominator == 0 {
        None
    } else {
        numerator.checked_div(denominator)
    }
}

/// Greatest Common Divisor using Euclidean algorithm
pub fn gcd(a: i32, b: i32) -> i32 {
    let mut a = a.abs();
    let mut b = b.abs();

    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }

    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Least Common Multiple
pub fn lcm(a: i32, b: i32) -> Option<i64> {
    if a == 0 || b == 0 {
        return Some(0);
    }
    let g = gcd(a, b) as i64;
    Some((a.abs() as i64 / g) * b.abs() as i64)
}

/// Check if number is prime
pub fn is_prime(n: i32) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Factorial with overflow handling
pub fn factorial(n: u32) -> Option<u64> {
    if n <= 1 {
        return Some(1);
    }

    let mut result: u64 = 1;
    for i in 2..=n as u64 {
        result = result.checked_mul(i)?;
    }
    Some(result)
}

/// Fibonacci with overflow handling
pub fn fibonacci(n: u32) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    if n == 1 {
        return Some(1);
    }

    let mut prev: u64 = 0;
    let mut curr: u64 = 1;

    for _ in 2..=n {
        let next = prev.checked_add(curr)?;
        prev = curr;
        curr = next;
    }
    Some(curr)
}

/// Power function with integer exponent
pub fn power(base: i64, exp: u32) -> Option<i64> {
    if exp == 0 {
        return Some(1);
    }
    if base == 0 {
        return Some(0);
    }

    let mut result: i64 = 1;
    let mut b = base;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = result.checked_mul(b)?;
        }
        b = b.checked_mul(b)?;
        e >>= 1;
    }
    Some(result)
}

/// Integer square root (floor)
pub fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }

    let mut x = n;
    let mut y = (x + 1) / 2;

    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Clamp value to range
pub fn clamp(value: i32, min_val: i32, max_val: i32) -> i32 {
    let (min_val, max_val) = if min_val > max_val {
        (max_val, min_val)
    } else {
        (min_val, max_val)
    };

    if value < min_val {
        min_val
    } else if value > max_val {
        max_val
    } else {
        value
    }
}

/// Linear interpolation
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    if t <= 0.0 {
        a
    } else if t >= 1.0 {
        b
    } else {
        a + (b - a) * t
    }
}

/// Calculate the sum of digits
pub fn digit_sum(mut n: u32) -> u32 {
    let mut sum = 0;
    while n > 0 {
        sum += n % 10;
        n /= 10;
    }
    sum
}

/// Check if number is a perfect square
pub fn is_perfect_square(n: u64) -> bool {
    if n == 0 {
        return true;
    }
    let root = isqrt(n);
    root * root == n
}

/// Calculate number of digits
pub fn num_digits(n: u64) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut count = 0;
    let mut num = n;
    while num > 0 {
        count += 1;
        num /= 10;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_add_basic() {
        assert_eq!(safe_add(2, 3), Some(5));
        assert_eq!(safe_add(-1, 1), Some(0));
    }
}
