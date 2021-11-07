import praw
from psaw import PushshiftAPI
import datetime as dt
import time
import re
import numpy as np
import pandas as pd

api = PushshiftAPI()

cid = "XAH_Zfot4yn9Jw"
cscrt = "ru-bwRQMvGGG5_7NEfL3we3R2qXB5A"
uname = "aplipala0"
pword = "0104Hsd30"
uagent = "prawNLP"

r = praw.Reddit(client_id=cid,client_secret=cscrt,username=uname,password=pword,user_agent=uagent)
api = PushshiftAPI(r)


start_epoch=int(dt.datetime(2018, 1, 1).timestamp())
end_epoch=int(dt.datetime(2019, 1, 1).timestamp())

k = list(api.search_submissions(after=start_epoch,
                            before = end_epoch,
                            subreddit='wallstreetbets'))
target_re = ["TSLA" , "AAPL"]
d = {"Date":[], "Title":[], "Up":[], "Down":[], "CountComments":[], "Award":[], "Uprate": [], "Crosspost":[]}

check_limit = 0
for subm in k:
    check_limit = check_limit + 1
    try:
        if not subm.stickied:
          time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(subm.created_utc))
          title = subm.title
          countup = str(subm.ups)
          countdown = str(subm.downs)
          numcom = str(subm.num_comments)
          award = str(subm.total_awards_received)
          uprate = str(subm.upvote_ratio)
          crosspost = str(subm.num_crossposts)


          print(title)
          print(countup)
          print(countdown)
          print(numcom)
          print(time_stamp[:10])
          print(award)
          print(subm.upvote_ratio)
          print(subm.num_crossposts)
          print(60*"-")
          if (int(numcom) != 0 and int(countup) != 0 and len(title) >= 30) or (award != 0):
            d["Date"].append(time_stamp[:10])
            d["Title"].append(title.strip())
            d["Up"].append(countup)
            d["Down"].append(countdown)
            d["CountComments"].append(numcom)
            d["Award"].append(award)
            d["Uprate"].append(uprate)
            d["Crosspost"].append(crosspost)
    except:
        pass




df = pd.DataFrame(data = d)
df.to_csv("2018_2019_reddit.csv", index=False)
#print(df)
#print(check_limit)
