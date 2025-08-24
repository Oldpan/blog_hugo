# 先git add 和 git commit

# 1) 确保 Pages 不跑 Jekyll
touch public/.nojekyll
# 如果有自定义域名，也把 CNAME 放进 public/
echo blog.oldpan.me > public/CNAME

hugo --minify
echo blog.oldpan.me > public/CNAME
git subtree split --prefix public -b gh-pages-tmp
git push -f origin gh-pages-tmp:gh-pages         
git branch -D gh-pages-tmp         