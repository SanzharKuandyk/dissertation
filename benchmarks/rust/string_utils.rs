//! String Utilities - Benchmark Rust Program
//! String manipulation functions for test generation evaluation

/// Check if string starts with prefix
pub fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

/// Check if string ends with suffix
pub fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

/// Count occurrences of character in string
pub fn count_char(s: &str, c: char) -> usize {
    s.chars().filter(|&ch| ch == c).count()
}

/// Count occurrences of substring
pub fn count_substring(s: &str, sub: &str) -> usize {
    if sub.is_empty() {
        return 0;
    }
    s.matches(sub).count()
}

/// Reverse a string
pub fn reverse_string(s: &str) -> String {
    s.chars().rev().collect()
}

/// Convert string to uppercase
pub fn to_uppercase(s: &str) -> String {
    s.to_uppercase()
}

/// Convert string to lowercase
pub fn to_lowercase(s: &str) -> String {
    s.to_lowercase()
}

/// Trim whitespace from both ends
pub fn trim(s: &str) -> &str {
    s.trim()
}

/// Check if string is palindrome
pub fn is_palindrome(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();

    if len <= 1 {
        return true;
    }

    for i in 0..len / 2 {
        if chars[i] != chars[len - 1 - i] {
            return false;
        }
    }
    true
}

/// Check if string is palindrome (case insensitive, alphanumeric only)
pub fn is_palindrome_normalized(s: &str) -> bool {
    let chars: Vec<char> = s
        .chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();

    let len = chars.len();
    if len <= 1 {
        return true;
    }

    for i in 0..len / 2 {
        if chars[i] != chars[len - 1 - i] {
            return false;
        }
    }
    true
}

/// Check if string contains only digits
pub fn is_numeric(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars().peekable();

    // Allow leading sign
    if let Some(&c) = chars.peek() {
        if c == '-' || c == '+' {
            chars.next();
            if chars.peek().is_none() {
                return false;
            }
        }
    }

    chars.all(|c| c.is_ascii_digit())
}

/// Check if string contains only alphabetic characters
pub fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic())
}

/// Check if string contains only alphanumeric characters
pub fn is_alnum(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric())
}

/// Find first occurrence of any character from set
pub fn find_first_of(s: &str, chars: &str) -> Option<usize> {
    s.find(|c| chars.contains(c))
}

/// Replace first occurrence of pattern
pub fn replace_first(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_string();
    }

    match s.find(from) {
        Some(pos) => {
            let mut result = String::with_capacity(s.len() - from.len() + to.len());
            result.push_str(&s[..pos]);
            result.push_str(to);
            result.push_str(&s[pos + from.len()..]);
            result
        }
        None => s.to_string(),
    }
}

/// Replace all occurrences of pattern
pub fn replace_all(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_string();
    }
    s.replace(from, to)
}

/// Split string into words (whitespace separated)
pub fn split_words(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

/// Count words in string
pub fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Capitalize first letter of string
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    }
}

/// Capitalize first letter of each word
pub fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| capitalize(word))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Check if string is empty or whitespace only
pub fn is_blank(s: &str) -> bool {
    s.trim().is_empty()
}

/// Truncate string to max length, adding ellipsis if needed
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        let mut result: String = s.chars().take(max_len - 3).collect();
        result.push_str("...");
        result
    }
}

/// Repeat string n times
pub fn repeat(s: &str, n: usize) -> String {
    s.repeat(n)
}

/// Check if two strings are anagrams
pub fn are_anagrams(s1: &str, s2: &str) -> bool {
    if s1.len() != s2.len() {
        return false;
    }

    let mut chars1: Vec<char> = s1.chars().collect();
    let mut chars2: Vec<char> = s2.chars().collect();

    chars1.sort_unstable();
    chars2.sort_unstable();

    chars1 == chars2
}

/// Levenshtein distance between two strings
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_string() {
        assert_eq!(reverse_string("hello"), "olleh");
        assert_eq!(reverse_string(""), "");
    }
}
