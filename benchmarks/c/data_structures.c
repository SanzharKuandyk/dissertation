/*
 * Data Structure Utilities - Benchmark C Program
 * Array and data structure operations for test generation evaluation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

// Find minimum value in array
int array_min(const int* arr, size_t size) {
    if (arr == NULL || size == 0) return INT_MAX;

    int min = arr[0];
    for (size_t i = 1; i < size; i++) {
        if (arr[i] < min) min = arr[i];
    }
    return min;
}

// Find maximum value in array
int array_max(const int* arr, size_t size) {
    if (arr == NULL || size == 0) return INT_MIN;

    int max = arr[0];
    for (size_t i = 1; i < size; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

// Calculate sum of array
long long array_sum(const int* arr, size_t size) {
    if (arr == NULL || size == 0) return 0;

    long long sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

// Calculate average of array
double array_avg(const int* arr, size_t size) {
    if (arr == NULL || size == 0) return 0.0;

    long long sum = array_sum(arr, size);
    return (double)sum / size;
}

// Find index of value in array (-1 if not found)
int array_find(const int* arr, size_t size, int value) {
    if (arr == NULL) return -1;

    for (size_t i = 0; i < size; i++) {
        if (arr[i] == value) return (int)i;
    }
    return -1;
}

// Binary search (array must be sorted)
int binary_search(const int* arr, size_t size, int value) {
    if (arr == NULL || size == 0) return -1;

    size_t left = 0;
    size_t right = size - 1;

    while (left <= right) {
        size_t mid = left + (right - left) / 2;

        if (arr[mid] == value) return (int)mid;
        if (arr[mid] < value) left = mid + 1;
        else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }
    return -1;
}

// Count occurrences of value in array
int array_count(const int* arr, size_t size, int value) {
    if (arr == NULL) return 0;

    int count = 0;
    for (size_t i = 0; i < size; i++) {
        if (arr[i] == value) count++;
    }
    return count;
}

// Check if array is sorted (ascending)
int is_sorted_asc(const int* arr, size_t size) {
    if (arr == NULL || size <= 1) return 1;

    for (size_t i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) return 0;
    }
    return 1;
}

// Check if array is sorted (descending)
int is_sorted_desc(const int* arr, size_t size) {
    if (arr == NULL || size <= 1) return 1;

    for (size_t i = 1; i < size; i++) {
        if (arr[i] > arr[i - 1]) return 0;
    }
    return 1;
}

// Reverse array in place
void array_reverse(int* arr, size_t size) {
    if (arr == NULL || size <= 1) return;

    size_t left = 0;
    size_t right = size - 1;

    while (left < right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
        left++;
        right--;
    }
}

// Rotate array left by k positions
void rotate_left(int* arr, size_t size, size_t k) {
    if (arr == NULL || size <= 1 || k == 0) return;

    k = k % size;
    if (k == 0) return;

    // Reverse first k elements
    array_reverse(arr, k);
    // Reverse remaining elements
    array_reverse(arr + k, size - k);
    // Reverse entire array
    array_reverse(arr, size);
}

// Remove duplicates from sorted array, returns new size
size_t remove_duplicates(int* arr, size_t size) {
    if (arr == NULL || size <= 1) return size;

    size_t write = 1;
    for (size_t read = 1; read < size; read++) {
        if (arr[read] != arr[read - 1]) {
            arr[write++] = arr[read];
        }
    }
    return write;
}

// Merge two sorted arrays into result (must be pre-allocated)
size_t merge_sorted(const int* arr1, size_t size1,
                   const int* arr2, size_t size2,
                   int* result) {
    if (result == NULL) return 0;
    if ((arr1 == NULL || size1 == 0) && (arr2 == NULL || size2 == 0)) return 0;

    if (arr1 == NULL || size1 == 0) {
        memcpy(result, arr2, size2 * sizeof(int));
        return size2;
    }
    if (arr2 == NULL || size2 == 0) {
        memcpy(result, arr1, size1 * sizeof(int));
        return size1;
    }

    size_t i = 0, j = 0, k = 0;

    while (i < size1 && j < size2) {
        if (arr1[i] <= arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }

    while (i < size1) result[k++] = arr1[i++];
    while (j < size2) result[k++] = arr2[j++];

    return k;
}

// Partition array around pivot, returns pivot index
size_t partition(int* arr, size_t size) {
    if (arr == NULL || size <= 1) return 0;

    int pivot = arr[size - 1];
    size_t i = 0;

    for (size_t j = 0; j < size - 1; j++) {
        if (arr[j] < pivot) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
        }
    }

    int temp = arr[i];
    arr[i] = arr[size - 1];
    arr[size - 1] = temp;

    return i;
}

// Check if two arrays are equal
int arrays_equal(const int* arr1, size_t size1,
                const int* arr2, size_t size2) {
    if (size1 != size2) return 0;
    if (arr1 == NULL && arr2 == NULL) return 1;
    if (arr1 == NULL || arr2 == NULL) return 0;

    for (size_t i = 0; i < size1; i++) {
        if (arr1[i] != arr2[i]) return 0;
    }
    return 1;
}

// Find second largest element
int second_largest(const int* arr, size_t size, int* result) {
    if (arr == NULL || size < 2 || result == NULL) return -1;

    int first = INT_MIN, second = INT_MIN;

    for (size_t i = 0; i < size; i++) {
        if (arr[i] > first) {
            second = first;
            first = arr[i];
        } else if (arr[i] > second && arr[i] != first) {
            second = arr[i];
        }
    }

    if (second == INT_MIN) return -1;  // No second largest

    *result = second;
    return 0;
}
