currentpath=$(pwd)
cd $paperspraypath

python scripts/add_papers.py

git add .
echo "Enter commit message" 
read commitmsg
git commit -m "$commitmsg"
git push origin master

cd $currentpath
