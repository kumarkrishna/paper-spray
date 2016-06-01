#!/usr/local/bin/python3

'''
Copies the readme template to the README.md file and appends the papers from
the json file.
'''

import json
import numpy as np


def main():
    with open('paper-list.json', 'r') as jsonfile:
        data = json.load(jsonfile)

    # Check for duplicates
    paper_names = set()
    for paper in data["data"]:
        paper_name = paper[2]
        chars = np.asarray(list(paper_name))
        ind1 = np.where(chars == '<')[0]
        ind2 = np.where(chars == '>')[0]
        try:
            assert ind1.shape[0] == 2 and ind2.shape[0] == 2
            ind1 = ind1[1]
            ind2 = ind2[0]+1
            paper_name = paper_name[ind2:ind1]
            paper_name = paper_name.lower()
            if paper_name in paper_names:
                print("Warning: Duplicate paper \"{}\"".format(paper_name))

            paper_names.add(paper_name)

        except AssertionError:
            pass

    with open('README.md', 'w') as readmefile:
        with open('readme.template', 'r') as templatefile:
            readmefile.write(templatefile.read())

        for item in data['data']:
	    readmefile.write('* {}  \n'.format(item[2]))
            readmefile.write('```{}, {}```  \n'.format(item[0], item[4]))
            readmefile.write('```{}, {}```  \n'.format(item[1], item[3]))

if __name__ == '__main__':
    main()
