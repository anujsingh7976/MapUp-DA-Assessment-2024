import os
import re
from typing import Any, Dict, List


#Question 2
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    grouped = {}
    for string in lst:
        length = len(string)
        if length not in grouped:
            grouped[length] = []
        grouped[length].append(string)
    sorted_grouped = dict(sorted(grouped.items()))

    return sorted_grouped
# Example
example1 = group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"])
example2 = group_by_length(["one", "two", "three", "four"])

print(example1)
print(example2)

#Question 3
def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    flattened = {}

    def flatten(current_dict: Dict[str, Any], parent_key: str = ''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                flatten(value, new_key)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        flatten(item, f"{new_key}[{index}]")
                    else:
                        flattened[f"{new_key}[{index}]"] = item
            else:
                flattened[new_key] = value

    flatten(nested_dict)
    return flattened

# Example usage
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)


#Question 4
def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(path: List[int], remaining: List[int], result: List[List[int]]):
        if not remaining:
            result.append(path)
            return
        # Use a set to track used numbers in this recursion level
        used = set()

        for i in range(len(remaining)):
            if remaining[i] in used:
                continue
            used.add(remaining[i])
            # Choose the current number and move forward
            backtrack(path + [remaining[i]], remaining[:i] + remaining[i+1:], result)

    result = []
    nums.sort()  # Sort to handle duplicates
    backtrack([], nums, result)
    return result

# Example usage
input_nums = [1, 1, 2]
permutations = unique_permutations(input_nums)
print(permutations)


#Question 5
def find_all_dates(text: str) -> List[str]:
    # Define regex patterns for the date formats
    patterns = [
        r'\b([0-2][0-9]|3[0-1])-(0[1-9]|1[0-2])-(\d{4})\b',    # dd-mm-yyyy
        r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b', # mm/dd/yyyy
        r'\b(\d{4})\.(0[1-9]|1[0-2])\.([0-2][0-9]|3[01])\b'      # yyyy.mm.dd
    ]

    # Combine patterns into one regex
    regex = re.compile('|'.join(patterns))

    # Find all matches in the text
    matches = regex.findall(text)

    # Create a list of full date strings
    dates = []
    for match in matches:
        # Check which format was matched and construct the full date string
        if match[0]:  # dd-mm-yyyy
            dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  # mm/dd/yyyy
            dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]:  # yyyy.mm.dd
            dates.append(f"{match[6]}.{match[7]}.{match[8]}")

    return dates

# Example usage
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
found_dates = find_all_dates(text)
print(found_dates)


#Question 6



#Question 7
def rotate_and_transform(matrix):
    """
    Rotates a square matrix by 90 degrees clockwise and transforms it.
    """
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Calculate the final transformed matrix
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Calculate sum of the row and column excluding the element itself
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - 2 * rotated_matrix[i][j]  # Adjusted calculation

    return final_matrix

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform(matrix)
print(result)  # Output should be [[22, 19, 16], [23, 20, 17], [24, 21, 18]]


#Question 8 start - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas as pd
import os

def check_timestamp_completeness(df: pd.DataFrame) -> pd.Series:
    # Create a multi-index based on (id, id_2)
    df.set_index(['id', 'id_2'], inplace=True)

    # Create a boolean series to hold results
    completeness_results = pd.Series(index=df.index.unique(), dtype=bool)

    # Iterate over each unique (id, id_2) pair
    for (id_val, id_2_val), group in df.groupby(level=[0, 1]):
        # Specify the format for datetime conversion
        datetime_format = '%Y-%m-%d %H:%M:%S'  # Adjust this format as needed

        # Convert timestamp columns to datetime, handling errors
        group['start'] = pd.to_datetime(group['startDay'] + ' ' + group['startTime'],
format=datetime_format,
errors='coerce')
        group['end'] = pd.to_datetime(group['endDay'] + ' ' + group['endTime'],
format=datetime_format,
errors='coerce')
        # Check for NaT values
        if group['start'].isnull().any() or group['end'].isnull().any():
            completeness_results[(id_val, id_2_val)] = False
            continue
        # Check if we have all days of the week (Monday=0, Sunday=6)
        days_covered = group['start'].dt.dayofweek.unique()
        complete_week = len(days_covered) == 7
        # Check if the timestamps cover a full 24-hour period
        full_day_covered = (group['end'].max() - group['start'].min()).days >= 1
        # Determine completeness for this (id, id_2) pair
        completeness_results[(id_val, id_2_val)] = complete_week and full_day_covered

    return completeness_results

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)  # Directory of the python.py
    data_path = os.path.join(current_dir, '..', 'datasets', 'dataset-1.csv')

    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_path)
    # Check completeness of timestamps
    results = check_timestamp_completeness(df)
    # Print the results
    print(results)



#Qestion 1
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    length = len(lst)
    # Process the list in chunks of size n
    for i in range(0, length, n):
        end = min(i + n, length)  # End index for the current group

        # Reverse the current group manually
        for j in range(end - 1, i - 1, -1):
            result.append(lst[j])

    return result

def main():
    current_dir = os.path.dirname(__file__)  # Directory of the python.py
    data_path = os.path.join(current_dir, '..', 'datasets', 'dataset-1.csv')

    columns_to_process = [
        'able2Hov2', 'able2Hov3', 'able3Hov2',
        'able3Hov3', 'able5Hov2', 'able5Hov3',
        'able4Hov2', 'able4Hov3'
    ]
    try:
        # Read the CSV file
        data = pd.read_csv(data_path)
        print("CSV file successfully connected.")
        # Process each row one by one
        for index, row in data.iterrows():
            # Get values of the specified columns in the current row
            row_values = [row[column] for column in columns_to_process if column in row]
            print(f"Row {index + 1} values: {row_values}")
            # Ask for 'n' value
            n = int(input("Enter the value of n for reversal (for this row): "))
            # Reverse the list by n elements
            reversed_list = reverse_by_n_elements(row_values, n)
            # Print the result for the current row
            print(f"Reversed list for Row {index + 1}: {reversed_list}\n")
    except FileNotFoundError:
        print(f"Error: {data_path} file not found.")
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
    except ValueError:
        print("Invalid input for 'n'. Please enter an integer.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
