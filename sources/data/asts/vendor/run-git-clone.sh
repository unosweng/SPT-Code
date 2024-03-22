while read repo; do
    git clone "$repo"
done < repolist.txt
