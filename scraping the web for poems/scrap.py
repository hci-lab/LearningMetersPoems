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

    print("number of poems in that page > ", len(poemsLinks))
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


def pullPoem(poem_url, bahr_name, file_name):
    '''
        * Parameter: the poem url
        * Function:  download the given poem and stores it in the {database}.
    '''

    # 1* Getting the shotor
    beautifulSoupObject = getBeautifulSoupObjectOfPage(poem_url)
    className = {"class": "bet-1"}
    thePoem = beautifulSoupObject.findAll("div", className)
    shotor = thePoem[0].findAll("h3")

    print("Number of shotor ", len(shotor))

    # 2* Get the Poet
    authorTag = beautifulSoupObject.find("meta", {"name": "author"})
    poet = authorTag.get("content").strip()
    print (poet)

    # 3* Building Abyat
    counter = 0
    abyat = []
    file = open(file_name, "a")
    while(counter < len(shotor) - 2):
        firstShatr = shotor[counter].text.strip()
        secondShatr = shotor[counter+1].text.strip()
        bayt = firstShatr + " " + secondShatr
        line = bayt + "," + secondShatr + "," + firstShatr + "," + bahr_name + "," + poet + "\n"
        file.write(line)
        abyat.append(bayt)
        counter += 2
    file.close()

    # testing
    print("Number of abyat ", len(abyat))
    for bayt in abyat:
        print(bayt)
###
# END pullPoem
###


def scrapBohor(file_nameCSV):
    BohorURLs = {"الطويل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D8%E6%ED%E1&Find=meaning",
                "الوافر": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E6%C7%DD%D1&Find=meaning",
                "البسيط": "https://www.aldiwan.net/poem.html?Word=%C7%E1%C8%D3%ED%D8&Find=meaning",
                "الكامل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%DF%C7%E3%E1&Find=meaning",
                "الرجز": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D1%CC%D2&Find=meaning",
                "الرمل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D1%E3%E1&Find=meaning",
                "السريع": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D3%D1%ED%DA&Find=meaning",
                "المنسرح": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E3%E4%D3%D1%CD&Find=meaning",
                "الخفيف": "https://www.aldiwan.net/poem.html?Word=%C7%E1%CE%DD%ED%DD&Find=meaning",
                "المجتث": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E3%CC%CA%CB&Find=meaning",
                "الخبب": "https://www.aldiwan.net/poem.html?Word=%C7%E1%CE%C8%C8&Find=meaning",
                "المتدارك": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E3%CA%CF%C7%D1%DF&Find=meaning",
    }

    file = open(file_nameCSV, "a+")
    b = "البيت"
    r = "الشطر الأيمن"
    l = "الشطر الأيسر"
    h = "البحر"
    p = "الشاعر"
    file.write(b + "," + l + "," + r + "," + h + "," + p + "\n")

    for bahr_name, bahr_url in BohorURLs.items():

        # 1* get all the peoms of that Bahr
        bahr_poems = getAllBaherPoemsPaths(bahr_url)

        # 2* pull the poems of that Bahr
        for poem in bahr_poems:
            poem_url = "https://www.aldiwan.net/" + poem
            pullPoem(poem_url, bahr_name, file_nameCSV)
    file.close()


# # #
scrapBohor("dataset.csv")
