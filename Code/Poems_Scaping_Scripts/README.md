## Description
This is a small script to scrap poems of بحور الشعر from  [موقع الدِّيوَان](https://www.aldiwan.net/)  into a CSV file.

## How to use the script?
In the `main.py` file,
create a dictionary as below.
>Please, use the same name used in the website.
```python
bohorLinks = {
   "الطويل": "https://www.aldiwan.net/poem.html?Word=%C7%E1%D8%E6%ED%E1&Find=meaning",
   "الوافر": "https://www.aldiwan.net/poem.html?Word=%C7%E1%E6%C7%DD%D1&Find=meaning"
}
```
Then, use `scrap.scrapBohor(bohorLinks, file_name)` to start scraping.
Pass the `dictionary` as the first parameter, and `file_name` as the second parameter.
> Name `csv file` as `firstName_secondName.csv`
```python
# The Scrap Function
scrap.scrapBohor(bohorLinks, "taha_magdy.csv")
```
Finally, run the script!
```bash
$ python3 main.py
```
## What's Next!!
Upload your `CSV file` to [Google Drive](https://drive.google.com/drive/folders/0B92iyATPP9xIa2RwUWVPVkFCNmc)


## CSV Format
**البيت**|**الشطر الأيسر**|**الشطر الأيمن**|**البحر**|**الشاعر**|
:-----:|:-----:|:-----:|:-----:|:-----:|
...|...|...|...|...|

Enjoy!
