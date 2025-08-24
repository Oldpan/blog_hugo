# 先git add 和 git commit

git subtree split --prefix public -b gh-pages-tmp
git push -f origin gh-pages-tmp:gh-pages         
git branch -D gh-pages-tmp         