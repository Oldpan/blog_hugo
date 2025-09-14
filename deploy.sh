#!/bin/bash
export https_proxy=http://127.0.0.1:7890
touch public/.nojekyll

hugo --minify
echo oldpan.me > public/CNAME
git subtree split --prefix public -b gh-pages-tmp
git push -f origin gh-pages-tmp:gh-pages         
git branch -D gh-pages-tmp         