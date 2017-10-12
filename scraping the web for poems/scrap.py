#!/usr/local/bin/python3
from urllib.request import urlopen
from bs4 import BeautifulSoup

# For Testing
tawel = "https://www.aldiwan.net/poem.html?\
Word=%C7%E1%D8%E6%ED%E1&Find=meaning"


def getBeautifulSoupObjectOfPage(link):
    '''
        * Parameter: A web link
        * Returns:   A BeautifulSoup Object for the page
    '''
    htmlPage = urlopen(link)
    return BeautifulSoup(htmlPage.read(), "html5lib")
###
# END getBeautifulSoupObjectOfPage
###
