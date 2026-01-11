/*
 * Math Utilities - Benchmark C Program
 * A collection of mathematical functions for test generation evaluation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

// Basic arithmetic with edge cases
int safe_add(int a, int b) {
    // Check for overflow
    if (a > 0 && b > INT_MAX - a) {
        return INT_MAX;  // Overflow protection
    }
    if (a < 0 && b < INT_MIN - a) {
        return INT_MIN;  // Underflow protection
    }
    return a + b;
}

int safe_multiply(int a, int b) {
    if (a == 0 || b == 0) return 0;

    // Check for overflow
    if (a > 0 && b > 0 && a > INT_MAX / b) return INT_MAX;
    if (a < 0 && b < 0 && a < INT_MAX / b) return INT_MAX;
    if (a > 0 && b < 0 && b < INT_MIN / a) return INT_MIN;
    if (a < 0 && b > 0 && a < INT_MIN / b) return INT_MIN;

    return a * b;
}

// Division with zero handling
int safe_divide(int numerator, int denominator, int* result) {
    if (denominator == 0) {
        return -1;  // Error: division by zero
    }
    if (numerator == INT_MIN && denominator == -1) {
        return -2;  // Error: overflow
    }
    *result = numerator / denominator;
    return 0;  // Success
}

// Greatest Common Divisor using Euclidean algorithm
int gcd(int a, int b) {
    a = abs(a);
    b = abs(b);

    if (a == 0) return b;
    if (b == 0) return a;

    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Least Common Multiple
long long lcm(int a, int b) {
    if (a == 0 || b == 0) return 0;
    int g = gcd(a, b);
    return ((long long)abs(a) / g) * abs(b);
}

// Check if number is prime
int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return 0;
        }
    }
    return 1;
}

// Factorial with overflow check
long long factorial(int n) {
    if (n < 0) return -1;  // Error: negative input
    if (n <= 1) return 1;

    long long result = 1;
    for (int i = 2; i <= n; i++) {
        // Check for overflow before multiplication
        if (result > LLONG_MAX / i) {
            return -2;  // Overflow
        }
        result *= i;
    }
    return result;
}

// Fibonacci with memoization (iterative for simplicity)
long long fibonacci(int n) {
    if (n < 0) return -1;
    if (n <= 1) return n;

    long long prev = 0, curr = 1;
    for (int i = 2; i <= n; i++) {
        long long next = prev + curr;
        if (next < curr) return -2;  // Overflow
        prev = curr;
        curr = next;
    }
    return curr;
}

// Power function with integer exponent
long long power(int base, int exp) {
    if (exp < 0) return 0;  // Not handling negative exponents
    if (exp == 0) return 1;
    if (base == 0) return 0;

    long long result = 1;
    long long b = base;

    while (exp > 0) {
        if (exp & 1) {
            result *= b;
        }
        b *= b;
        exp >>= 1;
    }
    return result;
}

// Integer square root (floor)
int isqrt(int n) {
    if (n < 0) return -1;
    if (n == 0) return 0;

    int x = n;
    int y = (x + 1) / 2;

    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

// Absolute value
int abs_val(int x) {
    if (x == INT_MIN) return INT_MAX;  // Handle overflow
    return x < 0 ? -x : x;
}

// Clamp value to range
int clamp(int value, int min_val, int max_val) {
    if (min_val > max_val) {
        // Swap if reversed
        int temp = min_val;
        min_val = max_val;
        max_val = temp;
    }
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// Linear interpolation (integer version)
int lerp(int a, int b, int t, int t_max) {
    if (t_max == 0) return a;
    if (t <= 0) return a;
    if (t >= t_max) return b;

    long long diff = (long long)b - a;
    return a + (int)((diff * t) / t_max);
}
