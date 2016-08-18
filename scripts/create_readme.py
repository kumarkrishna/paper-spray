'''
Copies the readme template to the README.md file and appends the papers from
the json file.
'''

import json
import io


def main():
    with open('paper-list.json', 'r') as jsonfile:
        data = json.load(jsonfile)

    with io.open('README.md', 'w', encoding='utf-8') as readmefile:
        with open('readme.template', 'r') as templatefile:
            readmefile.write(u''+templatefile.read())

        for item in data['data']:
            readmefile.write(u'* {}  \n'.format(item[2]))
            readmefile.write(u'```{}, {}```  \n'.format(item[0], item[4]))
            readmefile.write(u'```{}, {}```  \n'.format(item[1], item[3]))

if __name__ == '__main__':
    main()
