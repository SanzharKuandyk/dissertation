/**
 * string_utils.cpp — C++ string utility functions
 *
 * Uses std::string throughout.  All functions are free (non-member).
 * No templates, no operator overloads, no class methods.
 */

#include <string>
#include <cctype>
#include <cstddef>
#include <algorithm>

/**
 * Return the number of characters in s (same as s.size()).
 */
int string_length(const std::string& s) {
    return static_cast<int>(s.size());
}

/**
 * Return true if s starts with the given prefix.
 */
bool starts_with(const std::string& s, const std::string& prefix) {
    if (prefix.size() > s.size()) return false;
    return s.compare(0, prefix.size(), prefix) == 0;
}

/**
 * Return true if s ends with the given suffix.
 */
bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/**
 * Count occurrences of character ch in s.
 */
int count_char(const std::string& s, char ch) {
    int count = 0;
    for (char c : s) {
        if (c == ch) ++count;
    }
    return count;
}

/**
 * Count non-overlapping occurrences of sub in s.
 * Returns 0 if sub is empty.
 */
int count_substring(const std::string& s, const std::string& sub) {
    if (sub.empty()) return 0;
    int count = 0;
    std::size_t pos = 0;
    while ((pos = s.find(sub, pos)) != std::string::npos) {
        ++count;
        pos += sub.size();
    }
    return count;
}

/**
 * Return s with characters in reversed order.
 */
std::string reverse_string(const std::string& s) {
    return std::string(s.rbegin(), s.rend());
}

/**
 * Return a copy of s with all lowercase ASCII letters uppercased.
 */
std::string to_uppercase(const std::string& s) {
    std::string result = s;
    for (char& c : result) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return result;
}

/**
 * Return a copy of s with all uppercase ASCII letters lowercased.
 */
std::string to_lowercase(const std::string& s) {
    std::string result = s;
    for (char& c : result) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

/**
 * Return a copy of s with leading whitespace removed.
 */
std::string trim_left(const std::string& s) {
    std::size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start);
}

/**
 * Return a copy of s with trailing whitespace removed.
 */
std::string trim_right(const std::string& s) {
    std::size_t end = s.find_last_not_of(" \t\n\r");
    if (end == std::string::npos) return "";
    return s.substr(0, end + 1);
}

/**
 * Return a copy of s with leading and trailing whitespace removed.
 */
std::string trim(const std::string& s) {
    return trim_left(trim_right(s));
}

/**
 * Return true if s reads the same forwards and backwards (case-sensitive).
 * Empty string is considered a palindrome.
 */
bool is_palindrome(const std::string& s) {
    if (s.empty()) return true;
    std::size_t lo = 0, hi = s.size() - 1;
    while (lo < hi) {
        if (s[lo] != s[hi]) return false;
        ++lo; --hi;
    }
    return true;
}

/**
 * Return true if every character of s is a decimal digit ('0'..'9').
 * Returns false for an empty string.
 */
bool is_numeric(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

/**
 * Return true if every character of s is an ASCII letter.
 * Returns false for an empty string.
 */
bool is_alpha(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isalpha(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

/**
 * Return true if every character of s is an ASCII letter or digit.
 * Returns false for an empty string.
 */
bool is_alnum(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isalnum(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

/**
 * Count the number of whitespace-delimited words in s.
 */
int word_count(const std::string& s) {
    int count = 0;
    bool in_word = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            in_word = false;
        } else if (!in_word) {
            ++count;
            in_word = true;
        }
    }
    return count;
}

/**
 * Return a copy of s with every occurrence of from_ch replaced by to_ch.
 */
std::string replace_char(const std::string& s, char from_ch, char to_ch) {
    std::string result = s;
    for (char& c : result) {
        if (c == from_ch) c = to_ch;
    }
    return result;
}
