import json
import sys
import numpy as np
from datetime import datetime

if sys.version_info.major == 3:
    raw_input = input


def check_dateformat(inp):
    try:
        parsed = datetime.strptime(inp, "%d/%m/%Y")
        return datetime.strftime(parsed, "%d/%m/%Y")
    except Exception as e:
        print(e)
        print("Please enter date in the format DD/MM/YYYY")
        return None


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
            paper_name = paper_name.strip().lower()
            if paper_name in paper_names:
                print("Warning: Duplicate paper \"{}\"".format(paper_name))

            paper_names.add(paper_name)

        except AssertionError:
            pass

    papername = raw_input("Enter name of the paper\n")
    paperlink = raw_input("Enter link to the paper\n")
    reviewlink = raw_input("Review link? Enter 'n' if not\n")
    authorlist = raw_input("Enter the list of authors\n")
    conference = raw_input("Enter the name of the conference\n")

    dateadded = None
    while dateadded is None:
        dateadded = raw_input("Enter the date added\n")
        dateadded = check_dateformat(dateadded)

    keywords = raw_input("Keywords associated\n")

    if papername.strip().lower() in paper_names:
        print("\nWarning: A paper with the same name already exists.")
        decision = raw_input("Do you want to still add it? (y|n): ")
        while decision not in ['y', 'n']:
            decision = raw_input("Please enter 'y' or 'n': ")
        if decision == 'n':
            print("Paper not added\n")
            return

    print("Adding paper ...")

    newpaper = []
    newpaper.append(dateadded)
    newpaper.append(authorlist)
    newpaper.append("<a href=\""+paperlink+"\">"+papername+"</a>")
    newpaper.append(conference)
    newpaper.append(keywords)
    newpaper.append("<a href=\""+reviewlink+"\">"+"\[Review\]</a")

    data["data"] = [newpaper] + data["data"]
    data["data"] = sorted(data["data"], key=lambda x: datetime.strptime(x[0], '%d/%m/%Y'), reverse=True)

    for i in range(len(data["data"])):
        for j in range(len(data["data"][i])):
            data["data"][i][j] = data["data"][i][j].strip()

    with open('paper-list.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)

if __name__ == "__main__":
    main()
