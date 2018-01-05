"""This module contains Arabic tools for text analysis
"""

# letters.
hamza            = u'\u0621'
alef_mad         = u'\u0622'
alef_hamza_above = u'\u0623'
waw_hamza        = u'\u0624'
alef_hamza_below = u'\u0625'
yeh_hamza        = u'\u0626'
alef             = u'\u0627'
beh              = u'\u0628'
teh_marbuta      = u'\u0629'
teh              = u'\u062a'
theh             = u'\u062b'
jeem             = u'\u062c'
hah              = u'\u062d'
khah             = u'\u062e'
dal              = u'\u062f'
thal             = u'\u0630'
reh              = u'\u0631'
zain             = u'\u0632'
seen             = u'\u0633'
sheen            = u'\u0634'
sad              = u'\u0635'
dad              = u'\u0636'
tah              = u'\u0637'
zah              = u'\u0638'
ain              = u'\u0639'
ghain            = u'\u063a'
feh              = u'\u0641'
qaf              = u'\u0642'
kaf              = u'\u0643'
lam              = u'\u0644'
meem             = u'\u0645'
noon             = u'\u0646'
heh              = u'\u0647'
waw              = u'\u0648'
alef_maksura     = u'\u0649'
yeh              = u'\u064a'
madda_above      = u'\u0653'
hamza_above      = u'\u0654'
hamza_below      = u'\u0655'
alef_wasl        = u'\u0671'


tatweel          = u'\u0640'

# diacritics
fathatan         = u'\u064b'
dammatan         = u'\u064c'
kasratan         = u'\u064d'
fatha            = u'\u064e'
damma            = u'\u064f'
kasra            = u'\u0650'
shadda           = u'\u0651'
sukun            = u'\u0652'

# small letters
small_alef       = u"\u0670"
small_waw        = u"\u06e5"
small_yeh        = u"\u06e6"
#ligatures
lam_alef                     = u'\ufefb'
lam_alef_hamza_above         = u'\ufef7'
lam_alef_hamza_below         = u'\ufef9'
lam_alef_mad_above           = u'\ufef5'
simple_lam_alef              = u'\u0644\u0627'
simple_lam_alef_hamza_above  = u'\u0644\u0623'
simple_lam_alef_hamza_below  = u'\u0644\u0625'
simple_lam_alef_mad_above  = u'\u0644\u0622'

# Lists
alphabet = u''.join([
        alef, 
        beh,
        teh,
        theh,
        jeem,  
        hah,  
        khah, 
        dal,
        thal,
        reh,  
        zain,  
        seen, 
        sheen,  
        sad,  
        dad,  
        tah, 
        zah, 
        ain,
        ghain,  
        feh,  
        qaf,  
        kaf, 
        lam,  
        meem,  
        noon,  
        heh,  
        waw,  
        yeh, 
        hamza,   
        alef_mad,  
        alef_hamza_above,  
        waw_hamza,  
        alef_hamza_below,
        yeh_hamza, 
        alef_maksura,  
        teh_marbuta
        ])

tashkeel  = (fathatan,  dammatan,  kasratan, 
            fatha, damma, kasra, 
            sukun, 
            shadda)
harakat  = (  fathatan,    dammatan,    kasratan, 
            fatha,   damma,   kasra, 
            sukun
            )
shortharakat  = ( fatha,   damma,   kasra,  sukun)

tanwin  = (fathatan,   dammatan,    kasratan)

not_def_haraka = tatweel
liguatures = (
            lam_alef, 
            lam_alef_hamza_above, 
            lam_alef_hamza_below, 
            lam_alef_mad_above, 
            )
hamzat = (
            hamza, 
            waw_hamza, 
            yeh_hamza, 
            hamza_above, 
            hamza_below, 
            alef_hamza_below, 
            alef_hamza_above, 
            )
alefat = (
            alef, 
            alef_mad, 
            alef_hamza_above, 
            alef_hamza_below, 
            alef_wasl, 
            alef_maksura, 
            small_alef, 

        )
weak   = ( alef,  waw,  yeh,  alef_maksura)
yehlike =  ( yeh,   yeh_hamza,   alef_maksura,    small_yeh  )

wawLike   = ( waw,   waw_hamza,   small_waw )
tehLike   = ( teh,   teh_marbuta )

small   = ( small_alef,  small_waw,  small_yeh)
moon_letters = (hamza    , 
        alef_mad         , 
        alef_hamza_above , 
        alef_hamza_below , 
        alef             , 
        beh              , 
        jeem             , 
        hah              , 
        khah             , 
        ain              , 
        ghain            , 
        feh              , 
        qaf              , 
        kaf              , 
        meem             , 
        heh              , 
        waw              , 
        yeh
    )
sun_letters = (
        teh              , 
        theh             , 
        dal              , 
        thal             , 
        reh              , 
        zain             , 
        seen             , 
        sheen            , 
        sad              , 
        dad              , 
        tah              , 
        zah              , 
        lam              , 
        noon             , 
    )

"""
    * Some alphabet building tools
"""
def alphabet_excluding(excludedLetters):
    """returns the alphabet excluding the given letters.

    Args:
        excludedLetters (list['char']): letters to be excluded from the alphabet

    Returns:
        str: alphabet excluding the given excludedLetters

    Calling:
        print(alphabet_excluding([alef, beh, qaf, teh]))
        
    """
    return [x for x in alphabet if x not in excludedLetters]


def treat_as_the_same(listOfLetter, letter, text):
    """convert any letter in the `listOfLetter` to `letter` in the given text

    Args:
        listOfLetter (['chars'] or str) 
        letter (char)
        text (str)

    Returns:
        str: a text after changing all the `listOfLetter` to that char `letter`
    
    Example:
        line = 'قل أعوذ برب الناس'
        print(treat_as_the_same([alef_hamza_above], alef, line))
        print(treat_as_the_same([ain], qaf, line))
        
        
    """
    for x in listOfLetter:
        text = text.replace(x, letter)
    return text

file = open('mmm.txt', 'w')

file.write("|Name | Arabic Letter      \n<br>")
file.write("------------------|-------------------------------\n<br>")
file.write("|hamza            | \u0621 \n")
file.write("|alef_mad         | \u0622 \n")
file.write("|alef_hamza_above | \u0623 \n")
file.write("|waw_hamza        | \u0624 \n")
file.write("|alef_hamza_below | \u0625 \n")
file.write("|yeh_hamza        | \u0626 \n")
file.write("|alef             | \u0627 \n")
file.write("|beh              | \u0628 \n")
file.write("|teh_marbuta      | \u0629 \n")
file.write("|teh              | \u062a \n")
file.write("|theh             | \u062b \n")
file.write("|jeem             | \u062c \n")
file.write("|hah              | \u062d \n")
file.write("|khah             | \u062e \n")
file.write("|dal              | \u062f \n")
file.write("|thal             | \u0630 \n")
file.write("|reh              | \u0631 \n")
file.write("|zain             | \u0632 \n")
file.write("|seen             | \u0633 \n")
file.write("|sheen            | \u0634 \n")
file.write("|sad              | \u0635 \n")
file.write("|dad              | \u0636 \n")
file.write("|tah              | \u0637 \n")
file.write("|zah              | \u0638 \n")
file.write("|ain              | \u0639 \n")
file.write("|ghain            | \u063a \n")
file.write("|feh              | \u0641 \n")
file.write("|qaf              | \u0642 \n")
file.write("|kaf              | \u0643 \n")
file.write("|lam              | \u0644 \n")
file.write("|meem             | \u0645 \n")
file.write("|noon             | \u0646 \n")
file.write("|heh              | \u0647 \n")
file.write("|waw              | \u0648 \n")
file.write("|alef_maksura     | \u0649 \n")
file.write("|yeh              | \u064a \n")
file.write("|madda_above      | \u0653 \n<br>")
file.write("|hamza_above      | \u0654 \n<br>")
file.write("|hamza_below      | \u0655 \n<br>")
file.write("|alef_wasl        | \u0671 \n<br>")
file.write("--------------------------------------------------\n<br>")
file.close()
