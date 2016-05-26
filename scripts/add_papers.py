def write_to_file(instructions, newpaper, paperlist):
    with open('README.md', 'w') as filename :
        for line in instructions : 
            filename.write(line)
        for line in newpaper : 
            filename.write(line)
        for line in paperlist :
            filename.write(line)

def main() :
    currentfile = []
    with open('README.md', 'r') as filename : 
        for line in filename : 
            currentfile.append(line)
    indexlist = currentfile.index("## Papers\n")
    
    instructions = currentfile[0:indexlist+1]
    paperlist = currentfile[indexlist+1:]
    
    papername = raw_input("Enter name of the paper\n")
    paperlink = raw_input("Enter link to the paper\n")
    reviewlink = raw_input("Review link? Enter 'n' if not\n")
    authorlist = raw_input("Enter the list of authors\n")
    conference = raw_input("Enter the name of the conference\n")
    dateadded = raw_input("Enter the date added\n")
    keywords = raw_input("Keywords associated\n")
    newpaper = []
    if(reviewlink == 'n'):
	newpaper.append("* [" + papername + "](" + paperlink + ")  \n")
    else:
	newpaper.append("* [" + papername + "](" + paperlink + ") [ ``` (Review Article) ```](" + reviewlink + ")  \n")
    newpaper.append("``` " + dateadded + " , " + keywords + "```  \n" )
    newpaper.append("```"  + authorlist + " , " + conference + " ```  \n")
    
    write_to_file(instructions, newpaper, paperlist)

if __name__ == "__main__" :
    main()
            
