import os
import pandas as pd
import warnings

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Qestion1 9 start - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check if required columns are present
    required_columns = ['id_start', 'id_end', 'distance']
    for col in required_columns:
        if col not in df.columns:
            return pd.DataFrame()  # Return an empty DataFrame if a column is missing

    # Create a list of unique IDs
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel())

    # Create a DataFrame to hold the distance matrix
    distance_matrix = pd.DataFrame(0.0, index=unique_ids, columns=unique_ids)  # Use float for compatibility

    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        from_id = row['id_start']
        to_id = row['id_end']
        distance = row['distance']

        # Set the distance from A to B
        distance_matrix.at[from_id, to_id] += distance
        # Set the distance from B to A to ensure symmetry
        distance_matrix.at[to_id, from_id] += distance

    # Ensure the diagonal values are 0 (not strictly necessary as initialized)
    for id_ in unique_ids:
        distance_matrix.at[id_, id_] = 0.0  # Distance from id to itself is 0

    return distance_matrix
    

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)  # Directory of the python.py
    data_path = os.path.join(current_dir, '..', 'datasets', 'dataset-2.csv')

    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(data_path)
    
    # Print the resulting distance matrix
    print(distance_matrix)

#Qestion 9 end - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#Qestion 10 start - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    # Create an empty list to hold the unrolled data
    unrolled_data = []

    # Iterate over the rows and columns of the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip if the start and end IDs are the same
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                # Append the data to the list
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Convert the list to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example usage
if __name__ == "__main__":
    # Assuming distance_matrix is obtained from the previous question
    current_dir = os.path.dirname(__file__)  # Directory of the python.py
    data_path = os.path.join(current_dir, '..', 'datasets', 'dataset-2.csv')

    distance_matrix = calculate_distance_matrix(data_path)
    
    # Unroll the distance matrix
    unrolled_df = unroll_distance_matrix(distance_matrix)
    
    # Print the resulting unrolled DataFrame
    print(unrolled_df)

#Qestion 10 end - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#Qestion 11 start - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Print the actual column names for debugging
    print("Columns in the dataset:", df.columns.tolist())

    # Check if required columns are present
    required_columns = ['id_start', 'id_end', 'distance']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Ensure the distance column is of type float
    df['distance'] = df['distance'].astype(float)

    # Create a list of unique IDs
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel())

    # Create a DataFrame to hold the distance matrix
    distance_matrix = pd.DataFrame(0.0, index=unique_ids, columns=unique_ids)

    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        from_id = row['id_start']
        to_id = row['id_end']
        distance = row['distance']

        # Set the distance from A to B
        distance_matrix.at[from_id, to_id] += distance
        # Set the distance from B to A to ensure symmetry
        distance_matrix.at[to_id, from_id] += distance

    # Ensure the diagonal values are 0
    for id_ in unique_ids:
        distance_matrix.at[id_, id_] = 0.0  # Distance from id to itself is 0

    return distance_matrix

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)  # Directory of the python.py
    data_path = os.path.join(current_dir, '..', 'datasets', 'dataset-2.csv')

    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(data_path)
    
    # Print the resulting distance matrix
    print(distance_matrix)


#Qestion 11 end - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#Qestion 12 start - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def calculate_toll_rate(unrolled_df: pd.DataFrame) -> pd.DataFrame:
    # Define the toll rate coefficients
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates and add them as new columns
    for vehicle, rate in rate_coefficients.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * rate

    return unrolled_df

# Example usage
if __name__ == "__main__":
    # Assuming unrolled_df is obtained from the previous question
    current_dir = os.path.dirname(__file__)  # Directory of the python.py
    data_path = os.path.join(current_dir, '..', 'datasets', 'dataset-2.csv')

    distance_matrix = calculate_distance_matrix(data_path)
    unrolled_df = unroll_distance_matrix(distance_matrix)

    # Calculate toll rates
    toll_rates_df = calculate_toll_rate(unrolled_df)
    
    # Print the resulting DataFrame with toll rates
    print(toll_rates_df)


#Qestion 12 end - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



#Qestion 13 start - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import pandas as pd
import numpy as np
from datetime import time, timedelta

def calculate_time_based_toll_rates(toll_rates_df: pd.DataFrame) -> pd.DataFrame:
    # Define time intervals and discount factors
    discount_factors = {
        'weekday': {
            (time(0, 0), time(10, 0)): 0.8,
            (time(10, 0), time(18, 0)): 1.2,
            (time(18, 0), time(23, 59)): 0.8,
        },
        'weekend': 0.7
    }

    # Days of the week
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Initialize new columns
    results = []

    # Iterate through each unique (id_start, id_end) pair
    for (id_start, id_end), group in toll_rates_df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            for hour in range(24):
                start_time = time(hour, 0)
                end_time = time(hour, 59)
                vehicle_rates = {}
                
                # Calculate toll rates based on time and day
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    original_rate = group[vehicle].values[0]
                    
                    if day in ["Saturday", "Sunday"]:
                        # Apply weekend discount
                        discount = discount_factors['weekend']
                        vehicle_rates[vehicle] = original_rate * discount
                    else:
                        # Apply weekday discounts
                        for time_range, discount in discount_factors['weekday'].items():
                            if time_range[0] <= start_time < time_range[1]:
                                vehicle_rates[vehicle] = original_rate * discount
                                break  # Exit once the correct discount is found

                # Add the results for this time slot
                results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **vehicle_rates
                })

    # Create a DataFrame from the results
    final_df = pd.DataFrame(results)

    return final_df

# Example usage
if __name__ == "__main__":
    # Assuming toll_rates_df is obtained from the previous question
    current_dir = os.path.dirname(__file__)  # Directory of the python.py
    data_path = os.path.join(current_dir, '..', 'datasets', 'dataset-2.csv')

    distance_matrix = calculate_distance_matrix(data_path)
    unrolled_df = unroll_distance_matrix(distance_matrix)
    toll_rates_df = calculate_toll_rate(unrolled_df)

    # Calculate time-based toll rates
    time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rates_df)
    
    # Print the resulting DataFrame with time-based toll rates
    print(time_based_toll_rates_df)


#Qestion 13 end - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -