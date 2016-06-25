
set -e

git pull origin master

python scripts/add_papers.py
python scripts/create_readme.py

git commit -a -m "added paper"
git push origin master
