
# Buckwalter script 

### it use to solve **Arabic unicode problem** , it convert arabic characters to english characters like :-

#### **normal characters** :-

*  الالف   like **'ا'**  -->   **'A'**
* الباء   like **'ب'**  -->   **'b'**
* السين   like **'س'**  -->   **'s'**
* الميم   like **'م'**  -->   **'m'**
* الهاء   like **'ه'**  -->   **'h'**
*  الته   like **'ة'**  -->   **'t'**
* التاء   like **'ت'**  -->   **'t'**
* الشين   like **'ش'**  -->   **'$'**
#### ... etc

#### **characters that have diacritics** :-

* سين مشدودة   like **'سَّ'**  -->   **'s~'**

* ميم مضمومة   like **'مُ '**  -->   **'mu'**

* عين مفتوحة   like **'عَ'**  -->   **'Ea'**

*  لام مفتوحة   like **'لَ'**  -->   **'la'**

* كاف مضمومه   like **'كُ'**  -->   **'ku'**

*  ياء ساكنه   like **'يْ'**  -->   **'yo'**
#### ... etc


#### **some special charas** :-

*    الاف بدوم همذة   like **'ا'**  -->   **'A'**

* الاف بهمزة من فوق   like **'أ'**  -->   **'>'**

*  الاف بهمزة سفلية   like **'إ'**  -->   **'<'**

*            المدة   like **'ـ'**  -->   **'_'**
#### ... etc


## How to use :- 

* put the text that you want in **file** 
* use comand line to run script... using python 2 or 3    
```
python3 buckwalter.py -c file_that_have_text.txt [options]
```
* or using **jupyter notebook** 
```
%run path/buckwalter.py -c path/file_that_have_text.txt  [options]
```

### script options : -
* **-h**        : to show help menu
* **-hamza**    : to activate sensitive of **hamza** and distinguish between **'ا'** and **'أ'**
* **-madda**    : it consider as normal **'ا'** by defoult , so use this option to activate **madda** sensitive like **'آ'**
* **-t**        : to distinguish between **'ت'** and **'ة'**, and consider **'ة'** --> **'p'**
* **-harakat**  : to activate sensitive of **diacritics characters**
* **-tatweel**  : to activate sensitive of this think **'التـــــــــطويـــــــــل'**
* **-toUTF**    : convert the text to UTF-8

## Some examples :-

### we will use this text as example 
```
 الــــسلام علــيكم  ورَحْمَةُ اللهِ وَبَرَكاتُهُ
```


* #### it ignore diacritics characters and tatweel and madda and everything
> python3 buckwalter.py -c file.txt 

```
>> AlslAm Elykm  wrHmt Allh wbrkAth
```

* #### here we activate sensitive of diacritics characters and ignore anything else
> python3 buckwalter.py -c file.txt -harakat

```
>> AlslAm Elykm  wraHomatu Allhi wabarakAtuhu
```


* #### here we activate sensitive of diacritics characters and tatweel togther
> python3 buckwalter.py -c file.txt -harakat -tatweel

```
>> Al____slAm El__ykm  wraHomatu Allhi wabarakAtuhu
```


## Example on Some poems in Dataset :

> python3 buckwalter.py -c file.txt -harakat -tatweel -t

```
  وقَولي كَلَّما جَشأت وجاشَت مَكانَكِ تُحمَدِي أو تستريحي

>> wqawly kala~mA ja$At wjA$at makAnaki tuHmadiy Aw tstryHy



  فنَارُ القَلْبِ بَعْدَكُم تُصيِّرهُ على الثُلُثِ
  
>>  fnaAru Alqalobi baEodakum tuSyi~rhu Ely Alvuluvi


  لِمَن رَبعٌ بذاتِ الجيـ ـشِ أمسَى دارِساً خَلَقَا,ـشِ أمسَى دارِساً خَلَقَا

>> liman rabEN b*Ati Aljy_ _$i Amsay dArisAF xalaqaA,_$i Amsay dArisAF xalaqaA


```







## Review :- 
* this scrept used **Buckwalter Transliteration** technique to evaluate this idea 
* **Buckwalter Transliteration** use this technique to Voice Translator to can Arabic pronunciation

* Auther called : **Kenton Murray** @2014 [link](https://github.com/KentonMurray/Buckwalter)

