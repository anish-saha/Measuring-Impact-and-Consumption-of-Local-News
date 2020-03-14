# Script to parse all articles in xml format within the directory and extract metadata, @author: Anish Saha
import pandas as pd
import numpy as np
import os, glob, re
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

datetime_created = [] # xml tag: <datetime-created>
datetime_publish = [] # xml tag: <YMD>
newspaper = []        # xml tag: <ROY>
authors = []          # xml tag: <AUT>
headlines = []        # xml tag: <HED>
section = []          # xml tag: <SEC>
content = []          # xml tag: <p/>

for i in os.walk('../../../data/rhett'):
    subdirectory = i[0]
    for xml_file in Path(subdirectory).rglob('*.xml'):
        try:
            xml = ET.parse(xml_file)
            root = xml.getroot()
            
            temp = []
            for dt in root.iter('datetime-created'):
                temp.append(dt.text)
            datetime_created.append(' '.join(temp))
            
            temp = []
            for dt in root.iter('YMD'):
                temp.append(dt.text)
            datetime_publish.append(' '.join(temp))
            
            temp = []       
            for dt in root.iter('ROY'):
                temp.append(dt.text)
            newspaper.append(' '.join(temp))

            temp = []
            for dt in root.iter('AUT'):
                temp.append(dt.text)
            authors.append(' '.join(temp))
            
            temp = []
            for dt in root.iter('HED'):
                temp.append(dt.text)
            headlines.append(' '.join(temp))
            
            temp = []
            for dt in root.iter('SEC'):
                temp.append(dt.text)
            try: section.append(' '.join(temp))
            except: section.append(None)

            temp = []
            with open(xml_file, 'r') as x:
                text = x.readlines()[0]
                m = re.search('<p/>(.*)<p/>', text)
                if m: c = re.sub('<.*?>', '', m.group(1)) 
                else: c = ''
                if c: temp = c.split('<p/>')
                else: temp = []
            try: 
                content.append(''.join(temp))
            except:
                conent.append('NULL')
        
        except ET.ParseError as e:
            print(e)
            continue

#DEBUG
#print(authors[:20])
#print(headlines[:20])
#print(content[10])

df = pd.DataFrame(data={'date': datetime_created,
                        'datetime-publish': datetime_publish,
                        'newspaper': newspaper,
                        'author': authors,
                        'headlines': headlines,
                        'section': section,
                        'content (NEED TO PREPROCESS!)': content})
def process_datetime(x):
    string_dates = x.split(" ")
    dates = [datetime.strptime(i,"%Y-%m-%d") for i in string_dates]
    return min(dates).timestamp()

def process_date(x):
    x = x[:10]

def process_newspaper(x):
    x = x[:100]

df['datetime-publish'] = df['datetime-publish'].apply(process_datetime)
df['date'] = df['date'].apply(process_date)
df['newspaper'] = df['newspaper'].apply(process_newspaper)

df.to_csv('award_article_data.csv', sep='\t')
print('Article parsing complete.')

