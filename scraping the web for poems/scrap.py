#!/usr/local/bin/python3
'''
    # This file contains tools to:
        * get urls of the poems of the x Baher
        * pull poems into the database
    # Python 3 used unicode by default.
'''
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError

# For Testing
tawel = "https://www.aldiwan.net/poem.html?\
Word=%C7%E1%D8%E6%ED%E1&Find=meaning"


def getBeautifulSoupObjectOfPage(link):
    '''
        * Parameter: A web link
        > Returns:   A BeautifulSoup Object for the page
    '''
#    print("waiting for the server")
#    time.sleep(0.5)
    try:
        htmlPage = urlopen(link)
    except HTTPError:
        return None
    return BeautifulSoup(htmlPage.read(), "html5lib")
###
# END getBeautifulSoupObjectOfPage
###


def getAllPoemsPathsInOnePage(baherPageLink):
    '''
        * Parameter: The link of a page of a Baher
        > Returns:   A list of tha peoms links in that page
    '''

    poemsLinks = []

    try:
        beautifulSoupObject = getBeautifulSoupObjectOfPage(baherPageLink)
    except:
        return None
    poemsClassName = {"class": "record col-xs-12"}
    instanceList = beautifulSoupObject.findAll("div", poemsClassName)

    for element in instanceList:
        anchor = element.find("a")
        poemsLinks.append(anchor.get("href"))

    print("number of poems in that page > ", len(poemsLinks))
    return poemsLinks
###
# END getAllPoemsPathsInOnePage()
###


def getNumberOfPagesOfBaher(baherLink):
    '''
        * Parameter: The link of a Baher from aldiwan.net
        > Returns:   return the number of pages of that baher
    '''
    try:
        beautifulSoupObject = getBeautifulSoupObjectOfPage(baherLink)
    except:
        return 0

    try:
        pages = beautifulSoupObject.findAll("a", {"class": "wp_page_numbers"})
    except:
        return 0

    return len(pages)
###
# END getNumberOfPagesOfBaher
###


def getAllBaherPoemsPaths(baherLink):
    '''
        * Parameter: The link of the Baher from aldiwan.net
        > Returns:   A list of one page of the peoms links of the given baher.
    '''

    baherPoemsPaths = []

    # 1* Getting the number of pages in that Baher
    numberOfPages = getNumberOfPagesOfBaher(baherLink)

    # In case there are no pages
    if numberOfPages == 0:
        return []
    else:
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


def pullPoem(poem_url, bahr_name, file):
    '''
        * Parameter: the poem url
        * effect:  download the given poem and stores it in the {database}.
        * Returns: None in case of failure, 1 in case of success
    '''

    try:
        beautifulSoupObject = getBeautifulSoupObjectOfPage(poem_url)
    except:
        return None

    # *0 Check if the poem exists

    # 1* Getting the shotor
    className = {"class": "bet-1"}
    try:
        thePoem = beautifulSoupObject.findAll("div", className)
    except:
        return None
    shotor = thePoem[0].findAll("h3")

    # 2* Get the Poet
    try:
        authorTag = beautifulSoupObject.find("meta", {"name": "author"})
    except:
        return None
    poet = authorTag.get("content").strip()
    print (poet)

    # 3* Get el-no3
    href_value = "categories.html?Word=عامه&Find=wsf"
    try:
        no3Tag = beautifulSoupObject.find("a", {"href": href_value})
    except:
        return None
    no3 = "لا يوجد"
    if no3Tag is None:
        print("no3Tag ", no3Tag)
    else:
        # no3 = no3Tag.text()
        print("no3Tag ", no3Tag.string)
        no3 = no3Tag.string
        print("no3 ", no3)

    # 4* Get el-3asr
    # 5* Get era

    # 3* Building Abyat
    counter = 0
    abyat = []
    while(counter < len(shotor) - 1):
        firstShatr = shotor[counter].text.strip()
        secondShatr = shotor[counter+1].text.strip()
        bayt = firstShatr + " " + secondShatr
        line = bayt + "," + secondShatr + "," + firstShatr + "," + bahr_name + "," + poet + "," + no3 + "\n"
        file.write(line)
        abyat.append(bayt)
        counter += 2

    # testing
    print("Number of shotor ", len(shotor))
    print("Number of abyat ", len(abyat))
    for bayt in abyat:
        print(bayt)

    return 1
###
# END pullPoem
###


def scrapBohor(bohorLinks, file_nameCSV):
    '''
        * Parameter1: dictionary {"اسم البحر", "its url"}
    '''

    fileCSV = open(file_nameCSV, "a+")
    b = "البيت"
    r = "الشطر الأيمن"
    l = "الشطر الأيسر"
    h = "البحر"
    p = "الشاعر"
    n = "نوع القصيدة"

    # Saving ...
    fileCSV.write(b + "," + l + "," + r + "," + h + "," + p + "," + n + "\n")

    bahr_count = 0
    for bahr_name, bahr_url in bohorLinks.items():
        # 1* get all the peoms of that Bahr
        bahr_poems = getAllBaherPoemsPaths(bahr_url)

        length = len(bahr_poems)

        # I case there is no poems paths returned
        if length == 0:
            continue
        else:

            poem_counter = 1
            bahr_count += 1

            # 2* pull the poems of that Bahr
            for poem in bahr_poems:
                print("poem #", poem)
                print("poem ", poem_counter, "/", length)
                poem_counter += 1
                print("Baher _> [*]", bahr_count, " ", bahr_name)
                poem_url = "https://www.aldiwan.net/" + poem
                pullPoem(poem_url, bahr_name, fileCSV)

    fileCSV.close()


# # #
# Testing
#scrapBohor(bohorLinks, "dataset.csv")
# pullPoem("https://www.aldiwan.net/poem123.html", "بحر", "text.csv")
