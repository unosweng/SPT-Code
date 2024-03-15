#!/bin/sh


# Bold High Intensty
BIBlack="\033[1;90m"      # Black
BIRed="\033[1;91m"        # Red
BIGreen="\033[1;92m"      # Green
BIYellow="\033[1;93m"     # Yellow
BIBlue="\033[1;94m"       # Blue
BIPurple="\033[1;95m"     # Purple
BICyan="\033[1;96m"       # Cyan
BIWhite="\033[1;97m"      # White

# High Intensty backgrounds
On_IBlack="\033[0;100m"   # Black
On_IRed="\033[0;101m"     # Red
On_IGreen="\033[0;102m"   # Green
On_IYellow="\033[0;103m"  # Yellow
On_IBlue="\033[0;104m"    # Blue
On_IPurple="\033[10;95m"  # Purple
On_ICyan="\033[0;106m"    # Cyan
On_IWhite="\033[0;107m"   # White

BICyan='\033[1;96m'
COLOR_B='\033[1;90m'
BIGreen='\033[1;92m'
COLOR_D='\033[0m' # No Color
COLOR_E='\033[0;31m'

echo ${BICyan}Need to pull. ${COLOR_D}
git pull

# echo -e ${ACTION}Checking Git repo. ${COLOR_D}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo ${BICyan}Need to pull. ${COLOR_D}
    git pull
elif [[ `git status --porcelain` ]]; then
    echo ${BICyan}Need to commit and push. ${COLOR_D}
    now=$(date)
    make m="Updated at $now"
else
    git status
    echo ${BIGreen}Current branch is up to date with origin/master.${COLOR_D}
fi
echo ${BIYellow}Done.${COLOR_D}
echo

git status


