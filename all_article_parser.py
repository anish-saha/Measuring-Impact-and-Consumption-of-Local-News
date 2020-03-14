#script to parse all articles in xml format within the directory and extract metadata, @author: Anish Saha
import pandas as pd
import numpy as np
import os, glob, re
from pathlib import Path
from xml.etree import ElementTree as ET

date = []             # xml tag: <date>
ymd = []              # xml tag: <ymd>
newspaper = []        # xml tag: <paper>
section = []          # xml tag: <section>
authors = []          # xml tag: <author>
headlines = []        # xml tag: <mainttext>
content = []          # xml tag: <p/>
wordcnt = []          # xml tag: <word_count>

num_records = 3       # number of articles to retreive per xml file
num_articles = 0      # number of total articles retreived

# all articles
# xml_files = list(Path("../../../data/ready/").rglob("*.[xX][mM][lL]")) 
# subset of articles 
xml_files = list(Path("../../../data/ready/").rglob("*.[xX][mM][lL]"))
#print(xml_files)
print("Parsing articles...")
for xml_file in xml_files:
    num_records = 3
    if num_articles % 100 == 0:
        print(str(num_articles) + " records retrieved.")
    try:
        xml = ET.parse(xml_file)
        root = xml.getroot()
        num_records = min(num_records, sum(1 for _ in root.iter('date')))
        num_articles = num_articles + num_records        

        date_iter = root.iter('date')
        ymd_iter = root.iter('ymd')
        newspaper_iter = root.iter('paper')
        section_iter = root.iter('section')
        authors_iter = root.iter('author')
        headlines_iter = root.iter('maintext')
        wordcnt_iter = root.iter('word_count')
        while len(date) < num_articles:
            try: date.append(next(date_iter).text)
            except: date.append(None)
        while len(ymd) < num_articles:
            try: ymd.append(next(ymd_iter).text)
            except: ymd.append(None)
        while len(newspaper) < num_articles:
            try: newspaper.append(next(newspaper_iter).text)
            except: newspaper.append(None)
        while len(section) < num_articles:
            try: section.append(next(section_iter).text)
            except: section.append(None)
        while len(authors) < num_articles:
            try: authors.append(next(authors_iter).text)
            except: authors.append(None)
        while len(headlines) < num_articles:
            try: headlines.append(next(headlines_iter).text)
            except: headlines.append(None)

        temp_wordcnt = []
        while len(temp_wordcnt) < num_articles:
            try: temp_wordcnt.append(int(next(wordcnt_iter).text))
            except: temp_wordcnt.append(0)
        wordcnt.extend(temp_wordcnt)

        file_content = ''
        with open(xml_file, 'r') as x:
            text = x.readlines()[0]
            m = re.search('<p/>(.*)<p/>', text)
            if m: c = re.sub('<.*?>', '', m.group(1))
            else: c = ''
            if c: temp = c.split('<p/>')
            else: temp = ''
            try:
                file_content += str(' '.join(temp))
            except:
                file_conent += ''
        idx = 0
        text_data = file_content.split('\\s')
        for i in temp_wordcnt:
            while len(content) < num_articles:
                if i > 0:
                    curr = ' '.join(text_data[idx:(idx+i)]) 
                    content.append(curr)
                    idx = idx + i + 1
                else:
                    content.append(None)

    except ET.ParseError as e:
        print(e)
        continue

#print("=================\nDEBUG:\n=================")
#print(authors[:20])
#print(content[10])
#print(len(date))

df = pd.DataFrame(data={'date': date,
                        'datetime-publish': ymd,
                        'newspaper': newspaper,
                        'author': authors,
                        'headlines': headlines,
                        'section': section,
                        'content (NEED TO PREPROCESS!)': content})
df.to_csv('article_data_small.csv', sep='\t')
print('Article parsing complete.')

