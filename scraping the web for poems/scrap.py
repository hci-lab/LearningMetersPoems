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


def getAllPoemsPathsInOnePage(baherPageLink):
    '''
        * Parameter: The link of a page of a Baher
        * Returns:   A list of tha peoms links in that page
    '''

    poemsLinks = []

    beautifulSoupObject = getBeautifulSoupObjectOfPage(baherPageLink)
    poemsClassName = {"class": "record col-xs-12"}
    instanceList = beautifulSoupObject.findAll("div", poemsClassName)

    for element in instanceList:
        anchor = element.find("a")
        poemsLinks.append(anchor.get("href"))

    print("page > ", len(poemsLinks))
    return poemsLinks
###
# END getAllPeomsLinksOfOneBaher()
###


