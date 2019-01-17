#find * -type l -not -exec grep -q "^{}$" .gitignore \; -print >> .gitignore
git add .
git commit -m '$1' -a
git push origin master

