
git pull origin master

python scripts/add_papers.py
python scripts/create_readme.py

echo "Enter commit message"
git commit -a -m "added paper"
git push origin master
