def date_sorter():
    
    # Your code here
    # Full date
    global df
    dates_extracted = df.str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>\d?\d)[/|-](?P<year>\d{4}))')

    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>([0-2]?[0-9])|([3][01]))[/|-](?P<year>\d{2}))'))
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    del dates_extracted[3]
    del dates_extracted[4]
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<day>\d?\d) ?(?P<month>[a-zA-Z]{3,})\.?,? (?P<year>\d{4}))'))
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>[a-zA-Z]{3,})\.?-? ?(?P<day>\d\d?)(th|nd|st)?,?-? ?(?P<year>\d{4}))'))
    del dates_extracted[3]
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

    # Without day
    dates_without_day = df[index_left].str.extractall('(?P<origin>(?P<month>[A-Z][a-z]{2,}),?\.? (?P<year>\d{4}))')
    dates_without_day = dates_without_day.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d\d?)/(?P<year>\d{4}))'))
    dates_without_day['day'] = 1
    dates_extracted = dates_extracted.append(dates_without_day)
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

    # Only year
    dates_only_year = df[index_left].str.extractall(r'(?P<origin>(?P<year>\d{4}))')
    dates_only_year['day'] = 1
    dates_only_year['month'] = 1
    dates_extracted = dates_extracted.append(dates_only_year)
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

    # Year
    dates_extracted['year'] = dates_extracted['year'].apply(lambda x: '19' + x if len(x) == 2 else x)
    dates_extracted['year'] = dates_extracted['year'].apply(lambda x: str(x))

    # Month
    dates_extracted['month'] = dates_extracted['month'].apply(lambda x: x[1:] if type(x) is str and x.startswith('0') else x)
    month_dict = dict({'September': 9, 'Mar': 3, 'November': 11, 'Jul': 7, 'January': 1, 'December': 12,
                       'Feb': 2, 'May': 5, 'Aug': 8, 'Jun': 6, 'Sep': 9, 'Oct': 10, 'June': 6, 'March': 3,
                       'February': 2, 'Dec': 12, 'Apr': 4, 'Jan': 1, 'Janaury': 1,'August': 8, 'October': 10,
                       'July': 7, 'Since': 1, 'Nov': 11, 'April': 4, 'Decemeber': 12, 'Age': 8})
    dates_extracted.replace({"month": month_dict}, inplace=True)
    dates_extracted['month'] = dates_extracted['month'].apply(lambda x: str(x))

    # Day
    dates_extracted['day'] = dates_extracted['day'].apply(lambda x: str(x))

    # Cleaned date
    dates_extracted['date'] = dates_extracted['month'] + '/' + dates_extracted['day'] + '/' + dates_extracted['year']
    dates_extracted['date'] = pd.to_datetime(dates_extracted['date'])

    dates_extracted.sort_values(by='date', inplace=True)
    df1 = pd.Series(list(dates_extracted.index.labels[0]))
    
    return df1