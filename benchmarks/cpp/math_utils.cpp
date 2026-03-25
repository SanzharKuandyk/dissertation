/**
 * math_utils.cpp — C++ mathematical utility functions
 *
 * Design rules for parseability:
 *   - All functions are free (non-member) functions
 *   - No templates, no operator overloads
 *   - Simple, concrete parameter types (int, double, long long)
 *   - One function per logical unit
 */

#include <cstdint>
#include <climits>
#include <cmath>
#include <cstdlib>

/**
 * Return a + b, clamped to INT_MAX / INT_MIN on overflow.
 */
int safe_add(int a, int b) {
    if (b > 0 && a > INT_MAX - b) return INT_MAX;
    if (b < 0 && a < INT_MIN - b) return INT_MIN;
    return a + b;
}

/**
 * Return a * b, clamped to INT_MAX / INT_MIN on overflow.
 */
int safe_multiply(int a, int b) {
    if (a == 0 || b == 0) return 0;
    if (a > 0 && b > 0 && a > INT_MAX / b) return INT_MAX;
    if (a < 0 && b < 0 && a < INT_MAX / b) return INT_MAX;
    if ((a > 0 && b < 0 && b < INT_MIN / a) ||
        (a < 0 && b > 0 && a < INT_MIN / b)) return INT_MIN;
    return a * b;
}

/**
 * Divide numerator by denominator.
 * Returns 0 if denominator is zero.
 */
double safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}

/**
 * Greatest common divisor of two non-negative integers (Euclidean algorithm).
 */
int gcd(int a, int b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * Least common multiple of a and b.  Returns 0 if either is zero.
 */
int lcm(int a, int b) {
    if (a == 0 || b == 0) return 0;
    int g = gcd(a, b);
    return (a / g) * b;
}

/**
 * Return true if n is a prime number (n >= 2).
 */
bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/**
 * Compute n! iteratively.  Returns -1 on overflow (n > 12 for 32-bit int).
 */
int factorial(int n) {
    if (n < 0) return -1;
    if (n == 0 || n == 1) return 1;
    if (n > 12) return -1;  // overflow guard for int
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/**
 * Compute the n-th Fibonacci number iteratively (0-indexed).
 * Returns -1 if n < 0 or would overflow.
 */
int fibonacci(int n) {
    if (n < 0) return -1;
    if (n == 0) return 0;
    if (n == 1) return 1;
    int prev = 0, curr = 1;
    for (int i = 2; i <= n; i++) {
        int next = prev + curr;
        if (next < 0) return -1;  // overflow
        prev = curr;
        curr = next;
    }
    return curr;
}

/**
 * Raise base to the power exp (non-negative integer exponent).
 * Returns 1 for exp == 0.
 */
long long power(int base_val, int exp) {
    if (exp < 0) return 0;
    long long result = 1;
    long long b = static_cast<long long>(base_val);
    for (int i = 0; i < exp; i++) {
        result *= b;
    }
    return result;
}

/**
 * Integer square root: largest integer k such that k*k <= n.
 * Returns -1 for negative input.
 */
int isqrt(int n) {
    if (n < 0) return -1;
    if (n == 0) return 0;
    int k = static_cast<int>(std::sqrt(static_cast<double>(n)));
    while (k * k > n) --k;
    while ((k + 1) * (k + 1) <= n) ++k;
    return k;
}

/**
 * Absolute value of x (handles INT_MIN by returning INT_MAX).
 */
int abs_val(int x) {
    if (x == INT_MIN) return INT_MAX;
    return x < 0 ? -x : x;
}

/**
 * Clamp value to [lo, hi].
 */
int clamp(int value, int lo, int hi) {
    if (lo > hi) return value;
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

/**
 * Linear interpolation: returns a + (b - a) * t / t_max.
 * Returns a if t_max == 0.
 */
int lerp(int a, int b, int t, int t_max) {
    if (t_max == 0) return a;
    return a + (b - a) * t / t_max;
}

/**
 * Sum of decimal digits of n (uses absolute value).
 */
int digit_sum(int n) {
    if (n < 0) n = -n;
    int total = 0;
    while (n > 0) {
        total += n % 10;
        n /= 10;
    }
    return total;
}

/**
 * Return true if n is a perfect square (n >= 0).
 */
bool is_perfect_square(int n) {
    if (n < 0) return false;
    int root = isqrt(n);
    return root * root == n;
}
