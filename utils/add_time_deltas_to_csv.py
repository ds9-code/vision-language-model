"""
Script to add time_deltas column to existing CSVs with clinical features.
Computes intervals (in months) between consecutive scans from scandates.
"""

import pandas as pd
from datetime import datetime
import os


def parse_scandate(date_str):
    """
    Parse scandate string to datetime object or days integer.
    Handles both YYYYMMDD format (string) and days format (integer).
    """
    # Try to parse as YYYYMMDD date format
    try:
        if len(str(date_str)) == 8:
            return datetime.strptime(str(date_str), "%Y%m%d")
    except (ValueError, TypeError):
        pass
    
    # If not a valid date, treat as days (integer)
    return int(date_str)


def is_date_format(date_str):
    """
    Check if the scandate is in YYYYMMDD format (date) or days format (integer).
    Returns True if date format, False if days format.
    """
    try:
        date_val = str(date_str).strip()
        if len(date_val) == 8:
            datetime.strptime(date_val, "%Y%m%d")
            return True
    except (ValueError, TypeError):
        pass
    return False


def compute_time_deltas(scandates_str):
    """
    Compute time intervals between consecutive scans in months.
    
    Handles two formats:
    1. Date format: "20150304-20150624-20170222" (YYYYMMDD strings)
    2. Days format: "3038-3177" (integer days)
    
    Args:
        scandates_str: String like "20150304-20150624-20170222" or "3038-3177"
    
    Returns:
        String of intervals like "3,20" (months between scans)
    """
    scan_values = [str(s).strip() for s in str(scandates_str).split('-')]
    
    if len(scan_values) <= 1:
        return ""  # No intervals for single scan
    
    # Detect format from first value
    is_date = is_date_format(scan_values[0])
    
    intervals = []
    for i in range(1, len(scan_values)):
        if is_date:
            # Date format: parse as dates and compute difference
            date1 = datetime.strptime(scan_values[i-1], "%Y%m%d")
            date2 = datetime.strptime(scan_values[i], "%Y%m%d")
            delta_days = (date2 - date1).days
        else:
            # Days format: treat as integer days
            days1 = int(scan_values[i-1])
            days2 = int(scan_values[i])
            delta_days = days2 - days1
        
        # Convert days to months
        delta_months = round(delta_days / 30.44)  # Average days per month
        intervals.append(str(delta_months))
    
    return ','.join(intervals)


def add_time_deltas_to_csv(input_csv_path, output_csv_path):
    """
    Add time_deltas column to CSV.
    
    Args:
        input_csv_path: Path to input CSV
        output_csv_path: Path to output CSV with time_deltas column
    """
    print(f"Reading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    print(f"Computing time deltas for {len(df)} rows...")
    df['time_deltas'] = df['scandates'].apply(compute_time_deltas)
    
    print(f"Saving to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    
    print(f"Done! Added time_deltas column.")
    print(f"\nSample rows:")
    print(df[['pat_id', 'scandates', 'time_deltas', 'sex', 'resection_status']].head(10))
    
    # Print statistics
    num_scans = df['scandates'].apply(lambda x: len(x.split('-')))
    print(f"\nNumber of scans per patient:")
    print(num_scans.value_counts().sort_index())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add time_deltas column to CSV")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: File not found: {args.input_csv}")
        return
    
    add_time_deltas_to_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()

