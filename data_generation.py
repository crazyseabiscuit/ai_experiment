import pandas as pd
import argparse
import random

# List of web pages
web_pages = ["home", "about", "products", "services", "contact"]

# List of actions
actions = ["click", "scroll", "hover", "submit"]

# Generate random navigation activity
def generate_navigation_activity(num_activities):
    activities = []
    for _ in range(num_activities):
        page = random.choice(web_pages)
        action = random.choice(actions)
        timestamp = random.randint(1, 1000)
        activities.append((page, action, timestamp))
    return activities

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, help='Path of the output file')
parser.add_argument('--num_records', type=int, help='Number of records to generate')
args = parser.parse_args()
# Generate 10 random navigation activities
activity_list = generate_navigation_activity(args.num_records)

# Print the generated activity
for activity in activity_list:
    page, action, timestamp = activity
    print(f"User {action}ed on '{page}' at timestamp {timestamp}")


file_path = args.file_path
pd.DataFrame(activity_list).to_csv(file_path, index=False)
print(f"data file {file_path} is generated succefully!")
