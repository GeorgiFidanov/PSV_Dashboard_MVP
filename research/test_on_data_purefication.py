import pandas as pd
import json
from ast import literal_eval

# Load your CSV
df = pd.read_csv("twitter-competitor-profiles.csv", sep=';', low_memory=False)

# We'll store all extracted posts here
all_posts = []

for _, row in df.iterrows():
    try:
        posts_raw = row['posts']

        # Skip if posts column is missing or null
        if pd.isna(posts_raw) or posts_raw.strip() == "":
            continue

        # Some cells are stringified JSON; clean and parse them
        try:
            posts_list = json.loads(posts_raw)
        except:
            posts_list = literal_eval(posts_raw)  # fallback for malformed JSON

        # Iterate over each post and extract relevant info
        for post in posts_list:
            post_data = {
                "profile_id": row['id'],
                "profile_name": row['profile_name'],
                "followers": row['followers'],
                "post_id": post.get('post_id'),
                "description": post.get('description'),
                "date_posted": post.get('date_posted'),
                "likes": post.get('likes'),
                "views": post.get('views'),
                "reposts": post.get('reposts'),
                "replies": post.get('replies'),
                "hashtags": post.get('hashtags')
            }
            all_posts.append(post_data)

    except Exception as e:
        print("Error parsing row:", e)

# Convert all extracted posts into a clean DataFrame
posts_df = pd.DataFrame(all_posts)

# Save cleaned data
posts_df.to_csv("cleaned_posts.csv", index=False)

print("✅ Extracted", len(posts_df), "posts successfully!")

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

for _, row in posts_df.dropna(subset=['hashtags']).iterrows():
    hashtags = row['hashtags']
    if isinstance(hashtags, list):
        for tag in hashtags:
            G.add_edge(row['profile_name'], tag)

plt.figure(figsize=(12,8))
nx.draw(G, with_labels=True, node_size=800, font_size=8)
plt.title("Hashtag Network Between Competitors")
plt.show()




plt.scatter(posts_df['views'], posts_df['likes'])
plt.xlabel("Views")
plt.ylabel("Likes")
plt.title("Likes vs Views (All Posts)")
plt.show()


engagement = posts_df.groupby('profile_name')[['likes', 'views', 'reposts']].mean().sort_values('likes', ascending=False)
engagement.plot(kind='bar', figsize=(10,6))
plt.title("Average Engagement by Competitor Profile")
plt.show()


posts_df['date_posted'] = pd.to_datetime(posts_df['date_posted'])
posts_df.groupby(posts_df['date_posted'].dt.date)['post_id'].count().plot(figsize=(10,5))
plt.title("Posting Frequency Over Time")
plt.ylabel("Posts per Day")
plt.show()
