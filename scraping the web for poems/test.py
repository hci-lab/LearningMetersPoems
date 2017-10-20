#!/usr/local/bin/python3
'''
    # This file tests the scraping tootls
'''
import scraping_tools as tools

urls = []
urls.append("https://www.aldiwan.net/poem11984.html")
urls.append("https://www.aldiwan.net/poem1184.html")
urls.append("https://www.aldiwan.net/poem1124.html")

# Apply pullPoem over a list of urls.
x = 0
for url in urls:
    print (x)
    tools.pullPoem(url, "الطويل")
    x += 1
