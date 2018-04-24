
from __future__ import print_function
import freesound 
import os
import sys

#받은 api를 입력하세요.
api_key = 'API_KEY 입력'
folder = 'cry/' # folder to save

freesound_client = freesound.FreesoundClient()
freesound_client.set_token(api_key)

try:
    os.mkdir(folder)
except:
    pass


# Search Example
print("Searching for 'cry':")
print("----------------------------")

results_pager = freesound_client.text_search(
    query="cry",
    #filter="tag:tenuto duration:[0.0 TO 15.0]",
    sort="automatic_desc",
    fields="id,name,previews,username"
)
print("Num results:", results_pager.count)
print("\t----- PAGE 1 -----")


for sound in results_pager:
    try:
        print("\t-", sound.name, "by", sound.username)
        filename = sound.name.replace(u'"', '_')
        filename = sound.name.replace(u'.', '_')
        filename = sound.name.replace(u'/', '_')
        filename = sound.name.replace(u'*', '_')
        if not os.path.exists(folder + filename):
            sound.retrieve_preview(folder, filename)
    except:
        pass
        
for page_idx in range(results_pager.count):
    #print("\t----- PAGE {} -----".format())
    results_pager = results_pager.next_page()
    for sound in results_pager:
        try:
            print("\t-", sound.name, "by", sound.username)
            filename = sound.name.replace(u'"', '_')
            filename = sound.name.replace(u'.', '_')
            filename = sound.name.replace(u'*', '_')
            filename = sound.name.replace(u'/', '_')
                
            if not os.path.exists(folder + filename):
                sound.retrieve_preview(folder, filename)
        except:
            pass
    print()
    print("\n",page_idx,"\n")

