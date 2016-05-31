import json

def main() :

    with open('paper-list.json', 'r') as jsonfile:
        data = json.load(jsonfile)

    papername = raw_input("Enter name of the paper\n")
    paperlink = raw_input("Enter link to the paper\n")
    reviewlink = raw_input("Review link? Enter 'n' if not\n")
    authorlist = raw_input("Enter the list of authors\n")
    conference = raw_input("Enter the name of the conference\n")
    dateadded = raw_input("Enter the date added\n")
    keywords = raw_input("Keywords associated\n")
    newpaper = []
    newpaper.append(dateadded)
    newpaper.append(authorlist)
    newpaper.append("<a href=\""+paperlink+"\">"+papername+"</a>")
    newpaper.append(conference)
    newpaper.append(keywords)
    newpaper.append(reviewlink)
    
    data["data"] = [newpaper] + data["data"]
    
    with open('paper-list.json', 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__" :
    main()
            
