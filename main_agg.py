import pandas as pd
import sys

# Read the CSV file into a DataFrame
df = pd.read_csv(
    r"C:\Users\iao\Desktop\Landscape_Fråd_v1\DataTables_NO_2020_2020\main_table.csv"
)

# Create a new DataFrame with the specified columns
df_main = df[
    [
        "docdb_family_id",
        "appln_auth",
        "appln_nr",
        "appln_kind",
        "appln_filing_year",
        "appln_nr_epodoc",
        "appln_nr_original",
        "docdb_family_size",
        "receiving_office",
        "nb_applicants",
        "nb_inventors",
        "granted",
        "priority_auth",
        "main_ipc_group",
    ]
].copy()

# Create the application_number column
df_main["application_number"] = (
    df_main["appln_auth"] + df_main["appln_nr"].astype(str) + df_main["appln_kind"]
)

# Remove the appln_nr and appln_kind columns
df_main.drop(["appln_nr", "appln_kind"], axis=1, inplace=True)

# Rearrange columns to make application_number the second column
df_main = df_main[
    ["docdb_family_id", "application_number"]
    + [
        col
        for col in df_main.columns
        if col not in ["docdb_family_id", "application_number"]
    ]
]

# Aggregate all columns by docdb_family_id
agg_funcs = {
    "application_number": "first",  # Adjust this based on how you're aggregating
    "appln_auth": "first",  # or another function like max, depending on your needs
    "appln_filing_year": "first",
    "appln_nr_epodoc": lambda x: ", ".join(map(str, x.unique())),  # Convert to string
    "appln_nr_original": lambda x: ", ".join(map(str, x.unique())),  # Convert to string
    "docdb_family_size": "first",
    "receiving_office": "first",
    "nb_applicants": "first",
    "nb_inventors": "first",
    "granted": "first",
    "priority_auth": "first",
    "main_ipc_group": "first",
}

df_main_agg = df_main.groupby("docdb_family_id").agg(agg_funcs)

# Reset the index to make docdb_family_id a column again
df_main_agg.reset_index(inplace=True)


# Save the aggregated DataFrame to a new CSV file
output_file_path = (
    r"C:\Users\iao\Desktop\Landscape_Fråd_v1\DataTables_NO_2020_2020\main_table_agg.csv"
)
df_main_agg.to_csv(output_file_path, index=False)

# Display the aggregated DataFrame and confirmation message
print(df_main_agg)
print(f"Aggregated data saved to: {output_file_path}")

# Display the aggregated DataFrame
print(df_main_agg)
print(df_main_agg.head())
print(df_main_agg.info())

print(df_main_agg[["nb_applicants", "nb_inventors"]].head())
