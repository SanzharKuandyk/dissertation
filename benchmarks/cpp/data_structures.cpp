/**
 * data_structures.cpp — C++ array / vector utility functions
 *
 * All functions operate on raw int arrays (with explicit size) or
 * use simple scalar types.  No class methods, no templates, no STL
 * container parameters (keeps signatures unambiguous for the parser).
 */

#include <cstddef>
#include <climits>
#include <cmath>

/**
 * Return the minimum value in arr[0..n-1].
 * Returns INT_MAX if n <= 0.
 */
int array_min(const int* arr, int n) {
    if (n <= 0 || arr == nullptr) return INT_MAX;
    int min_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i];
    }
    return min_val;
}

/**
 * Return the maximum value in arr[0..n-1].
 * Returns INT_MIN if n <= 0.
 */
int array_max(const int* arr, int n) {
    if (n <= 0 || arr == nullptr) return INT_MIN;
    int max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    return max_val;
}

/**
 * Return the sum of arr[0..n-1].  Returns 0 for empty arrays.
 */
long long array_sum(const int* arr, int n) {
    if (n <= 0 || arr == nullptr) return 0LL;
    long long total = 0;
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }
    return total;
}

/**
 * Return the arithmetic mean of arr[0..n-1].
 * Returns 0.0 for empty arrays.
 */
double array_avg(const int* arr, int n) {
    if (n <= 0 || arr == nullptr) return 0.0;
    return static_cast<double>(array_sum(arr, n)) / n;
}

/**
 * Return the index of the first occurrence of target in arr[0..n-1].
 * Returns -1 if not found.
 */
int array_find(const int* arr, int n, int target) {
    if (n <= 0 || arr == nullptr) return -1;
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

/**
 * Binary search on a sorted array.  Returns the index of target,
 * or -1 if not found.  Behaviour is undefined if arr is not sorted.
 */
int binary_search(const int* arr, int n, int target) {
    if (n <= 0 || arr == nullptr) return -1;
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/**
 * Count occurrences of target in arr[0..n-1].
 */
int array_count(const int* arr, int n, int target) {
    if (n <= 0 || arr == nullptr) return 0;
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) ++count;
    }
    return count;
}

/**
 * Return true if arr[0..n-1] is sorted in non-decreasing order.
 * Empty and single-element arrays are considered sorted.
 */
bool is_sorted_asc(const int* arr, int n) {
    if (n <= 1 || arr == nullptr) return true;
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

/**
 * Return true if arr[0..n-1] is sorted in non-increasing order.
 */
bool is_sorted_desc(const int* arr, int n) {
    if (n <= 1 || arr == nullptr) return true;
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] < arr[i + 1]) return false;
    }
    return true;
}

/**
 * Reverse arr[0..n-1] in place.
 */
void array_reverse(int* arr, int n) {
    if (n <= 1 || arr == nullptr) return;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int tmp = arr[lo];
        arr[lo] = arr[hi];
        arr[hi] = tmp;
        ++lo; --hi;
    }
}

/**
 * Left-rotate arr[0..n-1] by k positions in place.
 */
void rotate_left(int* arr, int n, int k) {
    if (n <= 1 || arr == nullptr || k <= 0) return;
    k = k % n;
    if (k == 0) return;
    array_reverse(arr, k);
    array_reverse(arr + k, n - k);
    array_reverse(arr, n);
}

/**
 * Return the number of unique (deduplicated) elements when arr[0..n-1]
 * is treated as a set.  Counts distinct values, O(n^2) comparison.
 * Returns 0 for empty arrays.
 */
int count_unique(const int* arr, int n) {
    if (n <= 0 || arr == nullptr) return 0;
    int unique_count = 0;
    for (int i = 0; i < n; i++) {
        bool seen = false;
        for (int j = 0; j < i; j++) {
            if (arr[j] == arr[i]) { seen = true; break; }
        }
        if (!seen) ++unique_count;
    }
    return unique_count;
}

/**
 * Return the second largest distinct value in arr[0..n-1].
 * Returns INT_MIN if n < 2 or all values are equal.
 */
int second_largest(const int* arr, int n) {
    if (n < 2 || arr == nullptr) return INT_MIN;
    int first = INT_MIN, second = INT_MIN;
    for (int i = 0; i < n; i++) {
        if (arr[i] > first) {
            second = first;
            first = arr[i];
        } else if (arr[i] > second && arr[i] != first) {
            second = arr[i];
        }
    }
    return second;
}

/**
 * Return true if arr[0..n-1] contains target.
 */
bool array_contains(const int* arr, int n, int target) {
    return array_find(arr, n, target) != -1;
}

/**
 * Return the range (max - min) of arr[0..n-1].
 * Returns 0 for empty arrays.
 */
int array_range(const int* arr, int n) {
    if (n <= 0 || arr == nullptr) return 0;
    return array_max(arr, n) - array_min(arr, n);
}

/**
 * Compute the dot product of arr_a[0..n-1] and arr_b[0..n-1].
 * Returns 0 if either pointer is null or n <= 0.
 */
long long dot_product(const int* arr_a, const int* arr_b, int n) {
    if (n <= 0 || arr_a == nullptr || arr_b == nullptr) return 0LL;
    long long result = 0;
    for (int i = 0; i < n; i++) {
        result += static_cast<long long>(arr_a[i]) * arr_b[i];
    }
    return result;
}
