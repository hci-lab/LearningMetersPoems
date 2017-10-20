#!/usr/local/bin/python3
'''
'''
import scrap

bohorLinks = {"الطويل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D8%E6%ED%E1&Find=meaning",
            "الوافر": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E6%C7%DD%D1&Find=meaning",
            "البسيط": "https://www.aldiwan.net/poem.html?Word=%C7%E1%C8%D3%ED%D8&Find=meaning",
            "الكامل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%DF%C7%E3%E1&Find=meaning",
            "الرجز": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D1%CC%D2&Find=meaning",
            "الرمل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D1%E3%E1&Find=meaning",
#                "السريع": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D3%D1%ED%DA&Find=meaning",
#                "المنسرح": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E3%E4%D3%D1%CD&Find=meaning",
            "الخفيف": "https://www.aldiwan.net/poem.html?Word=%C7%E1%CE%DD%ED%DD&Find=meaning",
            "المجتث": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E3%CC%CA%CB&Find=meaning",
            "الخبب": "https://www.aldiwan.net/poem.html?Word=%C7%E1%CE%C8%C8&Find=meaning",
            "المتدارك": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E3%CA%CF%C7%D1%DF&Find=meaning",
}
scrap.scrapBohor(bohorLinks, "dataset.csv")
