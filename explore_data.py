#!/usr/bin/env python3
# explore_data.py - Place in project root directory

import pandas as pd
import os
from collections import Counter
import json

def explore_sekai_data():
    """Explore the structure and content of Sekai dataset"""
    
    data_dir = "data/raw"
    
    # 1. Load user data
    print("="*60)
    print("1. Exploring users.csv")
    print("="*60)
    users_df = pd.read_csv(os.path.join(data_dir, "users.csv"))
    print(f"Shape: {users_df.shape}")
    print(f"\nFirst 3 users' interest tags example:")
    for idx in range(min(3, len(users_df))):
        user_id = users_df.iloc[idx]['user_id']
        tags = users_df.iloc[idx]['user_interest_tags']
        print(f"\nUser {user_id}:")
        print(f"Tags: {tags[:150]}...")
    
    # Analyze tag distribution
    all_tags = []
    for tags_str in users_df['user_interest_tags'].dropna():
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        all_tags.extend(tags)
    
    tag_counter = Counter(all_tags)
    print(f"\nTotal unique tags: {len(tag_counter)}")
    print("\nTop 15 most common tags:")
    for tag, count in tag_counter.most_common(15):
        print(f"  {tag}: {count} times")
    
    # 2. Load content data
    print("\n" + "="*60)
    print("2. Exploring contents.csv")
    print("="*60)
    contents_df = pd.read_csv(os.path.join(data_dir, "contents.csv"))
    print(f"Shape: {contents_df.shape}")
    print(f"\nFirst 2 content examples:")
    for idx in range(min(2, len(contents_df))):
        content = contents_df.iloc[idx]
        print(f"\nContent ID: {content['content_id']}")
        print(f"Title: {content['title']}")
        print(f"Characters: {content['character_list']}")
        intro = str(content['intro'])[:200] if pd.notna(content['intro']) else "N/A"
        print(f"Intro preview: {intro}...")
    
    # 3. Load interaction data
    print("\n" + "="*60)
    print("3. Exploring interactions.csv")
    print("="*60)
    interactions_df = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    print(f"Shape: {interactions_df.shape}")
    print(f"\nInteraction statistics:")
    print(f"  Unique users: {interactions_df['user_id'].nunique()}")
    print(f"  Unique contents: {interactions_df['content_id'].nunique()}")
    print(f"  Interaction count statistics:")
    print(interactions_df['interaction_count'].describe())
    
    # 4. Data correlation analysis
    print("\n" + "="*60)
    print("4. Data Correlation Analysis")
    print("="*60)
    print(f"User coverage: {interactions_df['user_id'].nunique() / len(users_df) * 100:.1f}%")
    print(f"Content coverage: {interactions_df['content_id'].nunique() / len(contents_df) * 100:.1f}%")
    
    # Find most popular content
    popular_contents = interactions_df.groupby('content_id')['interaction_count'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(5) # type: ignore
    print(f"\nTop 5 most popular content (by total interactions):")
    for content_id, stats in popular_contents.iterrows():
        matching_contents = contents_df[contents_df['content_id'] == content_id]
        # content_title = matching_contents['title'].iloc[0] if len(matching_contents) > 0 else "Unknown"
        content_title = matching_contents['title'].values[0] if len(matching_contents) > 0 else "Unknown" # type: ignore
        print(f"  Content {content_id} ({content_title}): {stats['sum']} total interactions, {stats['count']} users")
    
    # Save data summary
    summary = {
        "users": {
            "total": len(users_df),
            "unique_tags": len(tag_counter),
            "top_tags": dict(tag_counter.most_common(10))
        },
        "contents": {
            "total": len(contents_df),
            "columns": list(contents_df.columns)
        },
        "interactions": {
            "total": len(interactions_df),
            "unique_users": interactions_df['user_id'].nunique(),
            "unique_contents": interactions_df['content_id'].nunique(),
            "avg_interaction_count": float(interactions_df['interaction_count'].mean())
        }
    }
    
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/data_summary.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nData summary saved to data/processed/data_summary.json")
    
    return users_df, contents_df, interactions_df

def analyze_data_quality_issues():
    """Analyze common data quality issues in the Sekai dataset"""
    
    data_dir = "data/raw"
    
    print("="*60)
    print("DATA QUALITY ISSUES ANALYSIS")
    print("="*60)
    
    # Load datasets
    users_df = pd.read_csv(os.path.join(data_dir, "users.csv"))
    contents_df = pd.read_csv(os.path.join(data_dir, "contents.csv"))
    
    # 1. Missing Values Analysis
    print("\n1. MISSING VALUES:")
    print("-" * 30)
    
    print("Users dataset missing values:")
    for col in users_df.columns:
        missing_count = users_df[col].isnull().sum()
        missing_pct = (missing_count / len(users_df)) * 100
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
    
    print("\nContents dataset missing values:")
    for col in contents_df.columns:
        missing_count = contents_df[col].isnull().sum()
        missing_pct = (missing_count / len(contents_df)) * 100
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
    
    # 2. Data Type Issues
    print("\n2. DATA TYPE ISSUES:")
    print("-" * 30)
    
    print("Users dataset dtypes:")
    for col, dtype in users_df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    print("\nContents dataset dtypes:")
    for col, dtype in contents_df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # 3. Inconsistent Data Formats
    print("\n3. INCONSISTENT DATA FORMATS:")
    print("-" * 30)
    
    # Check tag formats
    tag_lengths = []
    for tags_str in users_df['user_interest_tags'].dropna():
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        tag_lengths.append(len(tags))
    
    print(f"User interest tags per user:")
    print(f"  Min: {min(tag_lengths) if tag_lengths else 0}")
    print(f"  Max: {max(tag_lengths) if tag_lengths else 0}")
    print(f"  Avg: {sum(tag_lengths)/len(tag_lengths):.1f}" if tag_lengths else "N/A")
    
    # Check for empty or malformed tags
    empty_tags = 0
    malformed_tags = 0
    for tags_str in users_df['user_interest_tags'].dropna():
        if tags_str.strip() == '':
            empty_tags += 1
        elif tags_str.count(',') == 0 and len(tags_str.strip()) > 50:
            malformed_tags += 1
    
    print(f"  Empty tag strings: {empty_tags}")
    print(f"  Potentially malformed tags: {malformed_tags}")
    
    # 4. Duplicate Analysis
    print("\n4. DUPLICATE ANALYSIS:")
    print("-" * 30)
    
    user_duplicates = users_df.duplicated().sum()
    content_duplicates = contents_df.duplicated().sum()
    
    print(f"Duplicate users: {user_duplicates}")
    print(f"Duplicate contents: {content_duplicates}")
    
    # 5. Data Consistency Issues
    print("\n5. DATA CONSISTENCY ISSUES:")
    print("-" * 30)
    
    # Check for inconsistent character lists
    char_list_lengths = []
    for char_list in contents_df['character_list'].dropna():
        chars = [char.strip() for char in str(char_list).split(',') if char.strip()]
        char_list_lengths.append(len(chars))
    
    print(f"Character lists per content:")
    print(f"  Min: {min(char_list_lengths) if char_list_lengths else 0}")
    print(f"  Max: {max(char_list_lengths) if char_list_lengths else 0}")
    print(f"  Avg: {sum(char_list_lengths)/len(char_list_lengths):.1f}" if char_list_lengths else "N/A")
    
    # 6. Summary of Issues
    print("\n6. SUMMARY OF MAJOR ISSUES:")
    print("-" * 30)
    
    issues = []
    
    # Check for critical missing data
    if users_df['user_interest_tags'].isnull().sum() > len(users_df) * 0.1:
        issues.append("High percentage of missing user interest tags")
    
    if contents_df['intro'].isnull().sum() > len(contents_df) * 0.2:
        issues.append("High percentage of missing content introductions")
    
    # Check for data type mismatches
    if users_df['user_id'].dtype != 'int64':
        issues.append("User ID should be integer type")
    
    if contents_df['content_id'].dtype != 'int64':
        issues.append("Content ID should be integer type")
    
    # Check for empty datasets
    if len(users_df) == 0:
        issues.append("Users dataset is empty")
    
    if len(contents_df) == 0:
        issues.append("Contents dataset is empty")
    
    if issues:
        print("Critical issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No critical issues detected")
    
    print(f"\nTotal records - Users: {len(users_df)}, Contents: {len(contents_df)}")

def generate_data_quality_report():
    """Generate a comprehensive data quality report"""
    
    print("="*60)
    print("GENERATING DATA QUALITY REPORT")
    print("="*60)
    
    # Run the analysis
    analyze_data_quality_issues()
    
    # Save report to file
    report_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "analysis_type": "data_quality_issues",
        "recommendations": [
            "Implement data validation before processing",
            "Add missing value handling strategies",
            "Standardize tag and character list formats",
            "Consider data type conversions where appropriate",
            "Implement duplicate detection and removal"
        ]
    }
    
    with open("data_quality_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nReport saved to: data_quality_report.json")

if __name__ == "__main__":
    try:
        # Run the original exploration
        users_df, contents_df, interactions_df = explore_sekai_data()
        print("\n✅ Data exploration completed!")
        
        # Run the new data quality analysis
        print("\n" + "="*60)
        analyze_data_quality_issues()
        generate_data_quality_report()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Cannot find data files. Please ensure CSV files are in data/raw/ directory")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Expected data directory: {os.path.abspath('data/raw')}")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")