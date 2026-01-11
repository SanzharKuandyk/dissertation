/*
 * String Utilities - Benchmark C Program
 * String manipulation functions for test generation evaluation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Get string length (safe version)
size_t safe_strlen(const char* str) {
    if (str == NULL) return 0;
    return strlen(str);
}

// Compare strings (case insensitive)
int strcasecmp_safe(const char* s1, const char* s2) {
    if (s1 == NULL && s2 == NULL) return 0;
    if (s1 == NULL) return -1;
    if (s2 == NULL) return 1;

    while (*s1 && *s2) {
        int c1 = tolower((unsigned char)*s1);
        int c2 = tolower((unsigned char)*s2);
        if (c1 != c2) return c1 - c2;
        s1++;
        s2++;
    }
    return tolower((unsigned char)*s1) - tolower((unsigned char)*s2);
}

// Check if string starts with prefix
int starts_with(const char* str, const char* prefix) {
    if (str == NULL || prefix == NULL) return 0;

    while (*prefix) {
        if (*str != *prefix) return 0;
        str++;
        prefix++;
    }
    return 1;
}

// Check if string ends with suffix
int ends_with(const char* str, const char* suffix) {
    if (str == NULL || suffix == NULL) return 0;

    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);

    if (suffix_len > str_len) return 0;

    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

// Count occurrences of character in string
int count_char(const char* str, char c) {
    if (str == NULL) return 0;

    int count = 0;
    while (*str) {
        if (*str == c) count++;
        str++;
    }
    return count;
}

// Count occurrences of substring
int count_substring(const char* str, const char* sub) {
    if (str == NULL || sub == NULL || *sub == '\0') return 0;

    int count = 0;
    size_t sub_len = strlen(sub);
    const char* pos = str;

    while ((pos = strstr(pos, sub)) != NULL) {
        count++;
        pos += sub_len;
    }
    return count;
}

// Reverse string in place
void reverse_string(char* str) {
    if (str == NULL) return;

    size_t len = strlen(str);
    if (len <= 1) return;

    char* start = str;
    char* end = str + len - 1;

    while (start < end) {
        char temp = *start;
        *start = *end;
        *end = temp;
        start++;
        end--;
    }
}

// Convert string to uppercase (in place)
void to_uppercase(char* str) {
    if (str == NULL) return;

    while (*str) {
        *str = toupper((unsigned char)*str);
        str++;
    }
}

// Convert string to lowercase (in place)
void to_lowercase(char* str) {
    if (str == NULL) return;

    while (*str) {
        *str = tolower((unsigned char)*str);
        str++;
    }
}

// Trim leading whitespace (returns pointer within string)
char* trim_left(char* str) {
    if (str == NULL) return NULL;

    while (isspace((unsigned char)*str)) {
        str++;
    }
    return str;
}

// Trim trailing whitespace (modifies string)
void trim_right(char* str) {
    if (str == NULL) return;

    size_t len = strlen(str);
    while (len > 0 && isspace((unsigned char)str[len - 1])) {
        str[--len] = '\0';
    }
}

// Check if string is palindrome
int is_palindrome(const char* str) {
    if (str == NULL) return 0;

    size_t len = strlen(str);
    if (len <= 1) return 1;

    const char* left = str;
    const char* right = str + len - 1;

    while (left < right) {
        if (*left != *right) return 0;
        left++;
        right--;
    }
    return 1;
}

// Check if string contains only digits
int is_numeric(const char* str) {
    if (str == NULL || *str == '\0') return 0;

    // Allow leading minus sign
    if (*str == '-' || *str == '+') {
        str++;
        if (*str == '\0') return 0;
    }

    while (*str) {
        if (!isdigit((unsigned char)*str)) return 0;
        str++;
    }
    return 1;
}

// Check if string contains only alphabetic characters
int is_alpha(const char* str) {
    if (str == NULL || *str == '\0') return 0;

    while (*str) {
        if (!isalpha((unsigned char)*str)) return 0;
        str++;
    }
    return 1;
}

// Check if string contains only alphanumeric characters
int is_alnum(const char* str) {
    if (str == NULL || *str == '\0') return 0;

    while (*str) {
        if (!isalnum((unsigned char)*str)) return 0;
        str++;
    }
    return 1;
}

// Find first occurrence of any character from set
int find_first_of(const char* str, const char* chars) {
    if (str == NULL || chars == NULL) return -1;

    const char* p = str;
    while (*p) {
        const char* c = chars;
        while (*c) {
            if (*p == *c) return (int)(p - str);
            c++;
        }
        p++;
    }
    return -1;
}

// Replace first occurrence of old_char with new_char
int replace_char(char* str, char old_char, char new_char) {
    if (str == NULL) return 0;

    while (*str) {
        if (*str == old_char) {
            *str = new_char;
            return 1;  // Replaced
        }
        str++;
    }
    return 0;  // Not found
}

// Replace all occurrences of old_char with new_char
int replace_all_char(char* str, char old_char, char new_char) {
    if (str == NULL) return 0;

    int count = 0;
    while (*str) {
        if (*str == old_char) {
            *str = new_char;
            count++;
        }
        str++;
    }
    return count;
}
