#!/usr/local/bin/python3
'''
    # This file contains tools to:
        * get urls of the poems of the x Baher
        * pull poems into the database
    # Python 3 used unicode by default.
'''
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
# END getAllPoemsPathsInOnePage()
###


def getNumberOfPagesOfBaher(baherLink):
    '''
        * Parameter: The link of a Baher from aldiwan.net
        * Returns:   return the number of pages of that baher
    '''
    beautifulSoupObject = getBeautifulSoupObjectOfPage(baherLink)
    pages = beautifulSoupObject.findAll("a", {"class": "wp_page_numbers"})

    return len(pages)
###
# END getNumberOfPagesOfBaher
###



def getAllBaherPoemsPaths(baherLink):
    '''
        * Parameter: The link of the Baher from aldiwan.net
        * Returns:   A list of one page of the peoms links of the given baher.
    '''

    baherPoemsPaths = []

    # 1* Getting the number of pages in that Baher
    numberOfPages = getNumberOfPagesOfBaher(baherLink)

    # 2* apply the links of poems, page by page
    counter = 0
    while(counter <= numberOfPages):
        counter += 1
        # buliding the current page ulr
        currentPage = baherLink + "&Page=1"
        list = getAllPoemsPathsInOnePage(currentPage)
        baherPoemsPaths += list

    return baherPoemsPaths
###
# END getAllBaherPoemsPaths
###


def pullPoem(poem_url):
    '''
        * Parameter: the poem url
        * Function:  download the given poem and stores it in the database.
    '''

    # 1* Getting the shotor
    beautifulSoupObject = getBeautifulSoupObjectOfPage(poem_url)
    className = {"class": "bet-1"}
    thePoem = beautifulSoupObject.findAll("div", className)
    shotor = thePoem[0].findAll("h3")

    print(len(shotor))

    # 2* Building Abyat
    counter = 0
    abyat = []
    while(counter < len(shotor)):
        firstShatr = shotor[counter].text.strip()
        secondShatr = shotor[counter+1].text.strip()
        bayt = firstShatr + " " + secondShatr
        abyat.append(bayt)
        counter += 2


    # 3* Get the Poet
    authorTag = beautifulSoupObject.find("meta", {"name": "author"})
    poet = authorTag.get("content").strip()
    print (poet)


    # testing
    print(len(abyat))
    for bayt in abyat:
        print(bayt)
###
# END pullPoem
###

