import snscrape.modules.twitter as sntwitter
import pandas as pd

# ID del tweet de Bernie Sanders
tweet_id = "2005718422840303766"

# Lista para guardar datos
tweets_data = []

# Query para obtener TODA la conversación
query = f"conversation_id:{tweet_id}"

# Extraigo tweets
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    tweets_data.append([
        tweet.user.username,
        tweet.content,
        tweet.date,
        tweet.likeCount,
        tweet.inReplyToTweetId
    ])

# Creo DataFrame
df = pd.DataFrame(tweets_data, columns=[
    "username", "text", "date", "likes", "reply_to"
])

# Me quedo solo con comentarios (elimino el tweet original)
df = df[df["reply_to"].notna()]

# Guardo CSV
df.to_csv("data/raw/bernie_thread.csv", index=False)

print("Datos exportados correctamente")