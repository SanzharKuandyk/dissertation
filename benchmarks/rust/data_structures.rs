//! Data Structure Utilities - Benchmark Rust Program
//! Array and data structure operations for test generation evaluation

/// Find minimum value in slice
pub fn slice_min(arr: &[i32]) -> Option<i32> {
    arr.iter().cloned().min()
}

/// Find maximum value in slice
pub fn slice_max(arr: &[i32]) -> Option<i32> {
    arr.iter().cloned().max()
}

/// Calculate sum of slice
pub fn slice_sum(arr: &[i32]) -> i64 {
    arr.iter().map(|&x| x as i64).sum()
}

/// Calculate average of slice
pub fn slice_avg(arr: &[i32]) -> Option<f64> {
    if arr.is_empty() {
        return None;
    }
    Some(slice_sum(arr) as f64 / arr.len() as f64)
}

/// Find index of value in slice
pub fn find_index(arr: &[i32], value: i32) -> Option<usize> {
    arr.iter().position(|&x| x == value)
}

/// Binary search (slice must be sorted)
pub fn binary_search(arr: &[i32], value: i32) -> Option<usize> {
    arr.binary_search(&value).ok()
}

/// Count occurrences of value in slice
pub fn count_value(arr: &[i32], value: i32) -> usize {
    arr.iter().filter(|&&x| x == value).count()
}

/// Check if slice is sorted ascending
pub fn is_sorted_asc(arr: &[i32]) -> bool {
    arr.windows(2).all(|w| w[0] <= w[1])
}

/// Check if slice is sorted descending
pub fn is_sorted_desc(arr: &[i32]) -> bool {
    arr.windows(2).all(|w| w[0] >= w[1])
}

/// Reverse a vector in place
pub fn reverse_vec(arr: &mut Vec<i32>) {
    arr.reverse();
}

/// Rotate vector left by k positions
pub fn rotate_left(arr: &mut Vec<i32>, k: usize) {
    if arr.is_empty() || k == 0 {
        return;
    }
    let k = k % arr.len();
    arr.rotate_left(k);
}

/// Rotate vector right by k positions
pub fn rotate_right(arr: &mut Vec<i32>, k: usize) {
    if arr.is_empty() || k == 0 {
        return;
    }
    let k = k % arr.len();
    arr.rotate_right(k);
}

/// Remove duplicates from sorted slice, returns Vec with unique elements
pub fn remove_duplicates_sorted(arr: &[i32]) -> Vec<i32> {
    if arr.is_empty() {
        return Vec::new();
    }

    let mut result = vec![arr[0]];
    for &x in arr.iter().skip(1) {
        if x != *result.last().unwrap() {
            result.push(x);
        }
    }
    result
}

/// Merge two sorted slices into sorted Vec
pub fn merge_sorted(arr1: &[i32], arr2: &[i32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(arr1.len() + arr2.len());
    let mut i = 0;
    let mut j = 0;

    while i < arr1.len() && j < arr2.len() {
        if arr1[i] <= arr2[j] {
            result.push(arr1[i]);
            i += 1;
        } else {
            result.push(arr2[j]);
            j += 1;
        }
    }

    result.extend_from_slice(&arr1[i..]);
    result.extend_from_slice(&arr2[j..]);

    result
}

/// Partition slice around last element as pivot, returns pivot index
pub fn partition(arr: &mut [i32]) -> usize {
    if arr.len() <= 1 {
        return 0;
    }

    let pivot = arr[arr.len() - 1];
    let mut i = 0;

    for j in 0..arr.len() - 1 {
        if arr[j] < pivot {
            arr.swap(i, j);
            i += 1;
        }
    }

    arr.swap(i, arr.len() - 1);
    i
}

/// Check if two slices are equal
pub fn slices_equal(arr1: &[i32], arr2: &[i32]) -> bool {
    arr1 == arr2
}

/// Find second largest element
pub fn second_largest(arr: &[i32]) -> Option<i32> {
    if arr.len() < 2 {
        return None;
    }

    let mut first = i32::MIN;
    let mut second = i32::MIN;

    for &x in arr {
        if x > first {
            second = first;
            first = x;
        } else if x > second && x != first {
            second = x;
        }
    }

    if second == i32::MIN {
        None
    } else {
        Some(second)
    }
}

/// Find k-th smallest element (1-indexed)
pub fn kth_smallest(arr: &[i32], k: usize) -> Option<i32> {
    if k == 0 || k > arr.len() {
        return None;
    }

    let mut sorted = arr.to_vec();
    sorted.sort_unstable();
    Some(sorted[k - 1])
}

/// Calculate median
pub fn median(arr: &[i32]) -> Option<f64> {
    if arr.is_empty() {
        return None;
    }

    let mut sorted = arr.to_vec();
    sorted.sort_unstable();

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0)
    } else {
        Some(sorted[mid] as f64)
    }
}

/// Find mode (most frequent element)
pub fn mode(arr: &[i32]) -> Option<i32> {
    if arr.is_empty() {
        return None;
    }

    use std::collections::HashMap;
    let mut counts: HashMap<i32, usize> = HashMap::new();

    for &x in arr {
        *counts.entry(x).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
}

/// Find all pairs that sum to target
pub fn two_sum_pairs(arr: &[i32], target: i32) -> Vec<(i32, i32)> {
    use std::collections::HashSet;

    let mut seen: HashSet<i32> = HashSet::new();
    let mut result: Vec<(i32, i32)> = Vec::new();
    let mut used: HashSet<i32> = HashSet::new();

    for &x in arr {
        let complement = target - x;
        if seen.contains(&complement) && !used.contains(&x) && !used.contains(&complement) {
            let pair = if x < complement {
                (x, complement)
            } else {
                (complement, x)
            };
            result.push(pair);
            used.insert(x);
            used.insert(complement);
        }
        seen.insert(x);
    }

    result
}

/// Check if slice contains duplicate elements
pub fn has_duplicates(arr: &[i32]) -> bool {
    use std::collections::HashSet;
    let set: HashSet<_> = arr.iter().collect();
    set.len() != arr.len()
}

/// Calculate product of all elements
pub fn product(arr: &[i32]) -> Option<i64> {
    if arr.is_empty() {
        return None;
    }

    let mut result: i64 = 1;
    for &x in arr {
        result = result.checked_mul(x as i64)?;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_sum() {
        assert_eq!(slice_sum(&[1, 2, 3, 4, 5]), 15);
        assert_eq!(slice_sum(&[]), 0);
    }
}
