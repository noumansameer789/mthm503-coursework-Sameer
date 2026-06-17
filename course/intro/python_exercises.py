from datetime import date


def sum_list(values):
    """Return the sum of a list of numbers."""
    return sum(values)


def max_value(values):
    """Return the largest value in a list."""
    return max(values)


def reverse_string(text):
    """Return a string in reverse order."""
    return text[::-1]


def filter_even(values):
    """Return only the even numbers from a list."""
    return [value for value in values if value % 2 == 0]


def get_fifth_row(df):
    """Return the fifth row of a pandas DataFrame."""
    return df.iloc[4]


def column_mean(df, column):
    """Return the mean of a selected DataFrame column."""
    return df[column].mean()


def lookup_key(dictionary, key):
    """Return a dictionary value, or None if the key is missing."""
    return dictionary.get(key)


def count_occurrences(values):
    """Return a dictionary counting occurrences in a list."""
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def list_to_string(values):
    """Join a list of strings with commas."""
    return ",".join(values)


def parse_date(date_string):
    """Parse a YYYY-MM-DD string into a date object."""
    return date.fromisoformat(date_string)
