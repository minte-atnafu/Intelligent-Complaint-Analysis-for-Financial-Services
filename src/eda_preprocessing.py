
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Set up paths
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
INPUT_FILE = DATA_DIR / 'consumer_complaints.csv'
OUTPUT_FILE = DATA_DIR / 'filtered_complaints.csv'

# Load the dataset
df = pd.read_csv(INPUT_FILE, low_memory=False)

# --- Exploratory Data Analysis ---
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

# Analyze distribution of complaints across products
product_counts = df['Product'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=product_counts.values, y=product_counts.index)
plt.title('Distribution of Complaints by Product')
plt.xlabel('Number of Complaints')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig(DATA_DIR / 'product_distribution.png')
plt.close()

# Analyze narrative lengths
df['narrative_length'] = df['Consumer complaint narrative'].dropna().apply(lambda x: len(str(x).split()))
narrative_stats = df['narrative_length'].describe()
print("\nNarrative Length Statistics:\n", narrative_stats)

# Visualize narrative length distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['narrative_length'].dropna(), bins=50)
plt.title('Distribution of Narrative Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(DATA_DIR / 'narrative_length_distribution.png')
plt.close()

# Complaints with and without narratives
narrative_counts = {
    'With Narrative': df['Consumer complaint narrative'].notnull().sum(),
    'Without Narrative': df['Consumer complaint narrative'].isnull().sum()
}
print("\nNarrative Presence:\n", narrative_counts)

# --- Data Filtering ---
# Define target products
target_products = [
    'Credit card or prepaid card',
    'Consumer Loan',
    'Payday loan, title loan, or personal loan',
    'Money transfer, virtual currency, or money service',
    'Checking or savings account'
]

# Filter for target products and non-empty narratives
filtered_df = df[
    (df['Product'].isin(target_products)) & 
    (df['Consumer complaint narrative'].notnull())
].copy()

# Standardize product names for clarity
product_mapping = {
    'Credit card or prepaid card': 'Credit Card',
    'Consumer Loan': 'Personal Loan',
    'Payday loan, title loan, or personal loan': 'BNPL',
    'Money transfer, virtual currency, or money service': 'Money Transfers',
    'Checking or savings account': 'Savings Account'
}
filtered_df['Product'] = filtered_df['Product'].map(product_mapping)

# --- Text Cleaning ---
def clean_narrative(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove boilerplate phrases
    boilerplate = [
        r'i am writing to file a complaint',
        r'please help',
        r'xx/xx/xxxx',
        r'\s{2,}'  # Multiple spaces
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    # Strip extra whitespace
    text = text.strip()
    return text

# Apply cleaning
filtered_df['Consumer complaint narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_narrative)

# Remove any records where cleaning resulted in empty strings
filtered_df = filtered_df[filtered_df['Consumer complaint narrative'] != '']

# Save filtered dataset
filtered_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nFiltered dataset saved to {OUTPUT_FILE}")
print("Filtered Dataset Shape:", filtered_df.shape)



