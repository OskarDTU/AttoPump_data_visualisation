"""Quick script to inspect the experiment logs Excel file."""
import pandas as pd

EXCEL_PATH = (
    "/Users/oskarjohnbruunsmith/Library/CloudStorage/"
    "OneDrive-DanmarksTekniskeUniversitet/"
    "AttoPump - steps towards commercialisation/"
    "Tests + documentation/Logs/Experiment logs/"
    "Experiment logs_new format.xlsx"
)

df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
df = df.dropna(how="all")

print("Shape after dropping empty rows:", df.shape)
print("Columns:", list(df.columns))
print("---")
print(df.dtypes)
print("---")

tests = df[df["Test ID"].notna()]
print(f"Rows with Test ID: {len(tests)}")
print("---")

cols = ["Date", "Time", "Pump/BAR ID", "Test ID", "Test type", "Voltage", "Success/fail"]
print(tests[cols].head(30).to_string())
print("---")

print("Test type unique values:", tests["Test type"].unique())
print("---")

missing = tests[tests["Test type"].isna()]
print(f"Tests missing Test type: {len(missing)}")
if len(missing) > 0:
    print(missing[["Test ID", "Pump/BAR ID"]].to_string())
