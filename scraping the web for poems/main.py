#!/usr/local/bin/python3
'''
    * main script to pull poems from `aldiwan.net`

    * Create a dictionary {"اسم البحر": "url"}
    * Name the data set file as `your_name.csv`
'''
import scrap

bohorLinks = {
   "الطويل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D8%E6%ED%E1&Find=meaning",
   "الكامل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%DF%C7%E3%E1&Find=meaning"
}


# The Scrap Function
scrap.scrapBohor(bohorLinks, "taha_magdy.csv")
