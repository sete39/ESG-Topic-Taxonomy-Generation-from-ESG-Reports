import pandas as pd
import Reports
import re
from math import ceil
from copy import deepcopy
import pickle

print('Reading files...')
report_list = Reports.get_all_reports('../dataset/report_summary_extracted_data.jsonl')

print('Splitting into paragraphs...')
paragraph_list = []
paragraph_info_list = []
for report in report_list:
    for page in report.pages_list:
        sentence_list = []
        for block in page.block_list:
            # use text_str
            # remove all numbers
            # ignore if no full stop or comma
            # split by full stop
            # filtered_text = re.sub(r'\d+', '', block.text_str)
            sentence_list.append(block.text_str.strip())
        
        paragraph = ' '.join(sentence_list)
        paragraph_length = len(paragraph.split())
        if paragraph_length < 20:
            continue
        paragraph_list.append(paragraph)
        paragraph_info_list.append((page.block_list[0].page_count, report.url))
        

MAX_SENTENCE_LENGTH = 256
print('Splitting long paragraphs into max length of ', MAX_SENTENCE_LENGTH)
indices_to_pop = []
for i, p in enumerate(deepcopy(paragraph_list)):
    paragraph = p.split()
    length = len(paragraph)
    if i % 100000 == 0:
        print(i, len(paragraph_list))
    if length <= MAX_SENTENCE_LENGTH:
        continue
    
    n_splits = ceil(length/MAX_SENTENCE_LENGTH)
    split_paragraphs = [' '.join(paragraph[s*MAX_SENTENCE_LENGTH:(s*MAX_SENTENCE_LENGTH)+MAX_SENTENCE_LENGTH]) for s in range(n_splits)]
    split_paragraph_infos = [paragraph_info_list[i] for _ in range(n_splits)]
        
    indices_to_pop.append(i)
    
    paragraph_list.extend(split_paragraphs)
    paragraph_info_list.extend(split_paragraph_infos)
# need to reverse first so indices dont get messed up in the original array
for i in indices_to_pop[::-1]:
    paragraph_list.pop(i)
    paragraph_info_list.pop(i)

print('Final length: ', len(paragraph_list))

print('Saving...')
with open('../dataset/docs.pkl', 'wb') as handle:
    pickle.dump((paragraph_list, paragraph_info_list), handle, protocol=pickle.HIGHEST_PROTOCOL)