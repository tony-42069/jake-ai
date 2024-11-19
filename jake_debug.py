import json
import re
from pathlib import Path

def debug_tweets_file():
    # Use the exact path you provided
    tweets_file = Path(r"C:\Users\dsade\OneDrive\Desktop\Business\AI\jake_training\twitter-2024-11-19-25164d728f84b4ddcb46026d21bee8aee21e458263bfabfed68f4055c64d3c00\data\tweets.js")
    
    print("Starting tweet file analysis...")
    
    with open(tweets_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    print(f"\n1. File Stats:")
    print(f"- Total characters: {len(content)}")
    print(f"- Total lines: {content.count('\n') + 1}")
    
    print("\n2. First 200 characters of file:")
    print(content[:200])
    
    # Let's look at the structure
    cleaned_content = content.strip()
    if cleaned_content.startswith('window.YTD.tweets.part0 = '):
        cleaned_content = cleaned_content[25:]
    if cleaned_content.endswith(';'):
        cleaned_content = cleaned_content[:-1]
    
    print("\n3. Attempting to parse JSON...")
    try:
        tweets = json.loads(cleaned_content)
        print(f"SUCCESS! Found {len(tweets)} tweet objects!")
        
        # Look at the first tweet structure
        print("\n4. First tweet structure:")
        first_tweet = tweets[0]
        print(json.dumps(first_tweet, indent=2)[:500] + "...")
        
        # Count actual tweets
        valid_tweets = 0
        retweets = 0
        
        for tweet in tweets:
            if isinstance(tweet, dict) and 'tweet' in tweet:
                tweet_obj = tweet['tweet']
                text = tweet_obj.get('full_text', '') or tweet_obj.get('text', '')
                if text:
                    valid_tweets += 1
                    if text.startswith('RT @'):
                        retweets += 1
        
        print(f"\n5. Tweet Count Analysis:")
        print(f"- Total tweet objects: {len(tweets)}")
        print(f"- Valid tweets with text: {valid_tweets}")
        print(f"- Retweets: {retweets}")
        
    except json.JSONDecodeError as e:
        print(f"ERROR parsing JSON: {str(e)}")
        print("Location of error:")
        error_location = str(e).split('char')[-1].strip()
        try:
            char_pos = int(error_location)
            print(content[char_pos-50:char_pos+50])
            print("^--- Error around here")
        except:
            print("Couldn't locate exact error position")

if __name__ == '__main__':
    debug_tweets_file()