import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print(" Data cleaning started...")
df = pd.read_csv("C:\\Users\\prash\\Downloads\\Electric_Vehicle_Population_Data.csv")
print(df)
print("  Data imported successfully.")
print(" Initial number of rows: len(df)")
df_cleaned = df.drop_duplicates()
print("  Duplicates removed.")
print(" Rows after removing duplicates: len(df_cleaned)")
df_cleaned.columns = (
    df_cleaned.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("-", "_")
)
print(" Column names cleaned.")
before_drop = len(df_cleaned)
df_cleaned = df_cleaned.dropna(subset=["county", "city", "model_year", "make", "model"])
after_drop = len(df_cleaned)
print("  Dropped rows with missing values in key columns.")
print(" Rows removed: before_drop - after_drop")
print(" Rows remaining: after_drop")
df_cleaned['postal_code'] = df_cleaned['postal_code'].fillna(0).astype(int).astype(str)
df_cleaned['electric_range'] = df_cleaned['electric_range'].fillna(0)
df_cleaned['base_msrp'] = df_cleaned['base_msrp'].fillna(0)
df_cleaned['legislative_district'] = df_cleaned['legislative_district'].fillna(-1).astype(int)
df_cleaned['vehicle_location'] = df_cleaned['vehicle_location'].fillna("Unknown")
df_cleaned['electric_utility'] = df_cleaned['electric_utility'].fillna("Unknown")
df_cleaned['2020_census_tract'] = df_cleaned['2020_census_tract'].fillna(-1).astype(str)
print(" Step 5: Remaining missing values handled and data types converted.")
df_cleaned.to_csv("Cleaned_Electric_Vehicle_Data.csv", index=False)
print(" Cleaned data saved to 'Cleaned_Electric_Vehicle_Data.csv'.")
print(" Final cleaned data preview:")
print(df_cleaned.head())
print("Data cleaning completed successfully.")


print(" Data cleaning started...")


print(" Initial number of rows: {len(df)}")
df_cleaned = df.drop_duplicates()
print("  Duplicates removed.")
print(" Rows after removing duplicates: len(df_cleaned)")
df_cleaned.columns = (
    df_cleaned.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("-", "_")
)
print("Step 3: Column names cleaned.")
before_drop = len(df_cleaned)
df_cleaned = df_cleaned.dropna(subset=["county", "city", "model_year", "make", "model"])
after_drop = len(df_cleaned)
print("  Dropped rows with missing values in key columns.")
print(" Rows removed: before_drop - after_drop")
print(" Rows remaining: after_drop")
df_cleaned['postal_code'] = df_cleaned['postal_code'].fillna(0).astype(int).astype(str)
df_cleaned['electric_range'] = df_cleaned['electric_range'].fillna(0)
df_cleaned['base_msrp'] = df_cleaned['base_msrp'].fillna(0)
df_cleaned['legislative_district'] = df_cleaned['legislative_district'].fillna(-1).astype(int)
df_cleaned['vehicle_location'] = df_cleaned['vehicle_location'].fillna("Unknown")
df_cleaned['electric_utility'] = df_cleaned['electric_utility'].fillna("Unknown")
df_cleaned['2020_census_tract'] = df_cleaned['2020_census_tract'].fillna(-1).astype(str)
print("Step 5: Remaining missing values handled and data types converted.")
df_cleaned.to_csv("Cleaned_Electric_Vehicle_Data.csv", index=False)
print("  Cleaned data saved to 'Cleaned_Electric_Vehicle_Data.csv")
print(" Final cleaned data preview:")
print(df_cleaned.head())
print(df)
print("Showing the first 5 rows of the dataset:")
print(df.head())
print("Columns in the dataset:")
print(df.columns)



if 'model_year' in df.columns:
    print(" Analyzing number of EVs registered by model year...")
    year_counts = df['model_year'].value_counts().sort_index()
    print("Creating a line chart...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o")
    plt.title('Electric Vehicle Registrations by Model Year')
    plt.xlabel('Model Year')
    plt.ylabel('Number of Vehicles Registered')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Chart displayed successfully!")

else:
    print(" Column 'model_year' not found in the dataset. Please check the column names.")
print("Displaying the first few rows of the dataset to understand its structure:")
print(df.head())
if 'Make' in df.columns and 'Model' in df.columns:
    print("Both 'Make' and 'Model' columns are available for analysis.")
    print("Identifying the top 10 electric vehicle manufacturers...")
    top_makes = df['Make'].value_counts().head(10).reset_index()
    top_makes.columns = ['Make', 'Count']
    print("Plotting the top 10 EV makes...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_makes, x='Count', y='Make', hue='Make', legend=False, palette='viridis')
    plt.title('Top 10 Electric Vehicle Makes in Washington')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Vehicle Make')
    plt.tight_layout()
    plt.show()
    print("Bar chart for top 10 makes displayed!")
    print("Identifying the top 10 electric vehicle models...")
    top_models = df['Model'].value_counts().head(10).reset_index()
    top_models.columns = ['Model', 'Count']
    print("Plotting the top 10 EV models...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_models, x='Count', y='Model', hue='Model', legend=False, palette='mako')
    plt.title('Top 10 Electric Vehicle Models in Washington')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Vehicle Model')
    plt.tight_layout()
    plt.show()
    print("Bar chart for top 10 models displayed!")
else:
    print("Required columns 'Make' and/or 'Model' are missing in the dataset.")
if 'City' in df.columns:
    print("Found 'City' column, preparing charts...")
    top_cities = df['City'].value_counts().head(15) 
    # --- Pie Chart ---
    print("Creating pie chart...")
    plt.figure(figsize=(10, 6))
    plt.pie(top_cities, labels=top_cities.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 9})
    plt.title("Top 15 Cities with Most EV Registrations (Pie Chart)", fontsize=14, pad=15)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    print("Pie chart shown successfully!")
    # --- Bar Chart ---
    print("Creating bar chart...")
    top_cities_df = top_cities.reset_index()
    top_cities_df.columns = ['City', 'Count']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_cities_df, x='Count', y='City', hue='City', palette='cubehelix', legend=False)
    plt.title('Top 15 Cities with Most EV Registrations (Bar Chart)', fontsize=14)
    plt.xlabel('Number of Vehicles')
    plt.ylabel('City')
    plt.tight_layout()
    plt.show()
    print("Bar chart shown successfully!")

else:
    print("'City' column not found in the dataset.")
print("Analyzing growth trend by 'Model Year'...")
model_year_counts = df['Model Year'].value_counts().sort_index().reset_index()
model_year_counts.columns = ['Model Year', 'Count']
print("Vehicle count by model year calculated.")
print("Creating line chart...")
plt.figure(figsize=(12, 6))
sns.lineplot(data=model_year_counts, x='Model Year', y='Count', marker='o', color='teal')
plt.title('Electric Vehicle Registrations Over the Years', fontsize=14)
plt.xlabel('Model Year')
plt.ylabel('Number of Vehicles')
plt.grid(True)
plt.tight_layout()
plt.show()
print("Line chart displayed successfully!")
print("Objective 5: Analyze EV Types by Make (BEV vs PHEV) ---")
print("We are analyzing the distribution of Electric Vehicle Types (BEV and PHEV) across different car manufacturers (Makes).")
print("This helps us understand which companies are producing more Battery Electric Vehicles (BEVs) versus Plug-in Hybrid Electric Vehicles (PHEVs).\n")
make_type_counts = df.groupby(['Make', 'Electric Vehicle Type']).size().unstack(fill_value=0)
make_type_counts['Total'] = make_type_counts.sum(axis=1)
make_type_counts = make_type_counts.sort_values(by='Total', ascending=False).drop(columns='Total').head(10)
print("Top 10 EV Makes and their BEV/PHEV distribution:")
print(make_type_counts)
print("\nCreating stacked bar chart...")
make_type_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Paired')
plt.title('Top 10 EV Makes by Type (BEV vs PHEV)', fontsize=14)
plt.xlabel('Vehicle Make')
plt.ylabel('Number of Vehicles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Stacked bar chart displayed successfully!")
print("Objective 6: Analyze Electric Vehicle Electric Range ---")
print("We are analyzing how far electric vehicles can travel on a full charge, i.e., their 'Electric Range'.")
print("This will help us understand the common range brackets most EVs fall into.")
range_data = df['Electric Range'].dropna()
print("Electric Range Summary Statistics:")
print(range_data.describe())

# Create histogram plot
print("Creating distribution plot of electric vehicle ranges...")
plt.figure(figsize=(12, 6))
sns.histplot(range_data, bins=30, kde=True, color='skyblue')
plt.title('Distribution of Electric Vehicle Ranges (in miles)', fontsize=14)
plt.xlabel('Electric Range (miles)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
print("Histogram displayed successfully!")
# Select only numerical columns
numerical_columns = df.select_dtypes(include='number')

# Correlation matrix
correlation_matrix = numerical_columns.corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Correlation Heatmap - Electric Vehicle Data')
plt.tight_layout()
plt.show()
print("--- Objective 7: Explore Electric Vehicle Distribution by County ---")
print("We are identifying which counties in Washington have the highest number of electric vehicle registrations.")
print("This helps highlight regional EV adoption trends across the state.")

# Get top 10 counties by registration count
top_counties = df['County'].value_counts().head(10)
print("Top 10 Counties with Most EV Registrations:")
print(top_counties)

# Create a donut pie chart
print("Creating donut-style pie chart for county-wise EV distribution...")
plt.figure(figsize=(12, 6))
wedges, texts, autotexts = plt.pie(top_counties, labels=top_counties.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 9}, wedgeprops={'width': 0.4})
plt.title("Top 10 Counties by Electric Vehicle Registrations", fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
plt.tight_layout()
plt.show()
print("Pie chart displayed successfully!")
print(df.head())
print("Columns available:", df.columns)

# Boxplot for Electric Range
print("Creating boxplot for Electric Range...")
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['Electric Range'])
plt.title("Boxplot of Electric Vehicle Ranges")
plt.show()

# Detecting Outliers using IQR
print("Detecting outliers using IQR method...")
Q1 = df['Electric Range'].quantile(0.25)
Q3 = df['Electric Range'].quantile(0.75)
IQR = Q3 - Q1

print("Q1 (25th percentile):", Q1)
print("Q3 (75th percentile):", Q3)
print("IQR (Q3 - Q1):", IQR)

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower bound for outliers:", lower_bound)
print("Upper bound for outliers:", upper_bound)

# Identify Outliers
outliers = df[(df['Electric Range'] < lower_bound) | (df['Electric Range'] > upper_bound)]
print("Outliers detected:")
print(outliers[['Make', 'Model', 'Electric Range']])

print(f"Total number of outliers in 'Electric Range': {outliers.shape[0]}")

print("Manufacturer-wise Average Electric Range ---")
print("We're calculating the average electric range for each manufacturer.")
print("This helps us identify which EV brands offer longer driving distances on average.")
avg_range_by_make = df.groupby('Make')['Electric Range'].mean().sort_values(ascending=False).head(15).reset_index()

# Display the data
print("Top 15 Manufacturers by Average Electric Range:")
print(avg_range_by_make)

# Create horizontal lollipop chart
plt.figure(figsize=(12, 6))
plt.hlines(y=avg_range_by_make['Make'], xmin=0, xmax=avg_range_by_make['Electric Range'], color='skyblue')
plt.plot(avg_range_by_make['Electric Range'], avg_range_by_make['Make'], "o", color='teal')
plt.title('Top 15 Manufacturers by Average Electric Range', fontsize=14)
plt.xlabel('Average Electric Range (miles)')
plt.ylabel('Manufacturer')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---- Objective : Manufacturer-wise Average Electric Range ----
print("--- Objective 10: Manufacturer-wise Average Electric Range ---")
print("We're calculating the average electric range (in miles) for each manufacturer.")
print("This helps us identify which companies produce longer-range EVs on average.")

# Group by Make and calculate average range
avg_range_by_make = df.groupby('Make')['Electric Range'].mean().sort_values(ascending=False).head(15).reset_index()

# Display the top 15
print("Top 15 Manufacturers by Average Electric Range:")
print(avg_range_by_make)

# Create horizontal bar plot
print("Creating bar plot for average range per manufacturer...")
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_range_by_make, x='Electric Range', y='Make', hue='Make', palette='coolwarm', legend=False)
plt.title('Top 15 Manufacturers by Average Electric Range', fontsize=14)
plt.xlabel('Average Electric Range (miles)')
plt.ylabel('Manufacturer')
plt.tight_layout()
plt.show()
print("Bar plot displayed successfully!")

# Drop rows where 'Model Year' is missing
print("Cleaning model year data...")
df = df.dropna(subset=['Model Year'])

# Count vehicles by model year
model_year_counts = df['Model Year'].value_counts().sort_index().reset_index()
model_year_counts.columns = ['Model Year', 'Number of Vehicles']

# Display summary
print("Electric vehicle counts by model year:")
print(model_year_counts)

# Plotting using scatter plot
print("Generating scatter plot for EV growth over time...")
plt.figure(figsize=(12, 6))
sns.scatterplot(data=model_year_counts, x='Model Year', y='Number of Vehicles', color='purple', s=100)
plt.title('EV Registration Growth by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Number of Vehicles')
plt.grid(True)
plt.tight_layout()
plt.show()
print("Scatter plot displayed successfully!")




 

