import json


def main():
    with open('paper-list.json', 'r') as jsonfile:
        data = json.load(jsonfile)

    with open('README.md', 'w') as readmefile:
        with open('readme.template', 'r') as templatefile:
            readmefile.write(templatefile.read())

        for item in data['data']:
            readmefile.write('* {}  \n'.format(item[2]))
            readmefile.write('```{}, {}```  \n'.format(item[0], item[4]))
            readmefile.write('```{}, {}```  \n'.format(item[1], item[3]))

if __name__ == '__main__':
    main()
