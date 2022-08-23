import pandas as pd
import re

# cleaninf description of both train and test data
# delte html tag such as <li>

def cleaning(data, training):
    tmp_data = data.copy()
    clean_texts = []
    for text in tmp_data["description"]:
        # delete html tag
        text = remove_tag(text)
        # replace duble space with single space
        text = text.replace('  ', ' ')
        clean_texts.append(text)
    tmp_data["description"] = clean_texts
    if training:
        tmp_data["jobflag"] += -1
    return tmp_data

def remove_tag(x):
    p = re.compile(r"<[^>]*?>")
    return p.sub(' ',x)