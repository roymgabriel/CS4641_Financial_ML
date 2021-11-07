import twint

# # Configure
# c = twint.Config()
# c.Username = "JeffBezos"
# # c.Search = "great"
# c.Store_csv = True
# c.Output = "Trump.csv"
# # Run
# twint.run.Search(c)

c = twint.Config()

c.Username = "elonmusk"
c.Custom["tweet"] = ["id", "username", "created_at", "username", "mentions" , "replies_count", "retweets_count", "likes_count", "hashtags", "tweet"]
c.Pandas = True

# c.Output = c.Username

twint.run.Search(c)

Tweets_df = twint.storage.panda.Tweets_df

print(Tweets_df)