import pandas as pd
import os

# 设置数据路径
DATA_PATH = "data/raw"

def explore_data():
    """探索三个CSV文件的内容和结构"""
    
    print("="*80)
    print("SEKAI数据集探索")
    print("="*80)
    
    # 1. 探索users.csv
    print("\n1. USERS.CSV 探索:")
    print("-"*40)
    users_df = pd.read_csv(os.path.join(DATA_PATH, "users.csv"))
    print(f"总用户数: {len(users_df)}")
    print(f"列名: {list(users_df.columns)}")
    print("\n前5个用户样例:")
    print(users_df.head())
    
    # 分析用户标签
    print("\n用户兴趣标签分析:")
    all_tags = []
    for tags in users_df['user_interest_tags'].dropna():
        all_tags.extend([tag.strip() for tag in tags.split(',')])
    unique_tags = set(all_tags)
    print(f"总共有 {len(unique_tags)} 个不同的标签")
    print(f"最常见的20个标签:")
    from collections import Counter
    tag_counter = Counter(all_tags)
    for tag, count in tag_counter.most_common(20):
        print(f"  - {tag}: {count}次")
    
    # 2. 探索contents.csv
    print("\n\n2. CONTENTS.CSV 探索:")
    print("-"*40)
    contents_df = pd.read_csv(os.path.join(DATA_PATH, "contents.csv"))
    print(f"总内容数: {len(contents_df)}")
    print(f"列名: {list(contents_df.columns)}")
    print("\n前3个内容样例:")
    for idx in range(min(3, len(contents_df))):
        print(f"\n内容 {idx+1}:")
        print(f"  ID: {contents_df.iloc[idx]['content_id']}")
        print(f"  标题: {contents_df.iloc[idx]['title']}")
        print(f"  简介: {contents_df.iloc[idx]['intro'][:200]}...")
        print(f"  角色列表: {contents_df.iloc[idx]['character_list']}")
        print(f"  初始记录: {contents_df.iloc[idx]['initial_record'][:200]}...")
    
    # 3. 探索interactions.csv
    print("\n\n3. INTERACTIONS.CSV 探索:")
    print("-"*40)
    interactions_df = pd.read_csv(os.path.join(DATA_PATH, "interactions.csv"))
    print(f"总交互记录数: {len(interactions_df)}")
    print(f"列名: {list(interactions_df.columns)}")
    print("\n交互统计:")
    print(f"平均交互次数: {interactions_df['interaction_count'].mean():.2f}")
    print(f"最大交互次数: {interactions_df['interaction_count'].max()}")
    print(f"最小交互次数: {interactions_df['interaction_count'].min()}")
    
    # 分析用户活跃度
    user_interactions = interactions_df.groupby('user_id').agg({
        'content_id': 'count',
        'interaction_count': 'sum'
    }).rename(columns={'content_id': 'content_count', 'interaction_count': 'total_interactions'})
    
    print(f"\n用户活跃度分析:")
    print(f"平均每个用户互动的内容数: {user_interactions['content_count'].mean():.2f}")
    print(f"平均每个用户的总交互次数: {user_interactions['total_interactions'].mean():.2f}")
    
    # 分析内容受欢迎度
    content_popularity = interactions_df.groupby('content_id').agg({
        'user_id': 'count',
        'interaction_count': 'sum'
    }).rename(columns={'user_id': 'user_count', 'interaction_count': 'total_interactions'})
    
    print(f"\n内容受欢迎度分析:")
    print(f"平均每个内容的用户数: {content_popularity['user_count'].mean():.2f}")
    print(f"平均每个内容的总交互次数: {content_popularity['total_interactions'].mean():.2f}")
    
    # 保存数据的基本统计信息
    print("\n\n保存数据统计信息到 data/processed/data_stats.txt...")
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/data_stats.txt", "w", encoding='utf-8') as f:
        f.write("SEKAI数据集统计信息\n")
        f.write("="*50 + "\n")
        f.write(f"用户数: {len(users_df)}\n")
        f.write(f"内容数: {len(contents_df)}\n")
        f.write(f"交互记录数: {len(interactions_df)}\n")
        f.write(f"独特标签数: {len(unique_tags)}\n")
        f.write(f"平均每用户交互内容数: {user_interactions['content_count'].mean():.2f}\n")
        f.write(f"平均每内容被交互用户数: {content_popularity['user_count'].mean():.2f}\n")

if __name__ == "__main__":
    explore_data()