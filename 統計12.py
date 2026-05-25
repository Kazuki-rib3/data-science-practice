import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

tables = pd.read_html(url, storage_options={"User-Agent": headers["User-Agent"]})
df = tables[2]
print(df.head(10))
df.to_csv("gdp_ranking.csv", index=False)
print("CSVに保存しました")