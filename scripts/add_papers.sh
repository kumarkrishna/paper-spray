currentpath=$(pwd)
cd $paperspraypath

git pull origin master
python scripts/add_papers.py
python scripts/create_readme.py

git add .
echo "Enter commit message" 
read commitmsg
git commit -m "$commitmsg"
git push origin master

cd $currentpath
