# Check if DATE is provided as a command-line argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <date> (e.g., ./tail.sh 1225-1100)"
    exit 1
fi

DATE="$1"

tail -f output-spt-code-2024-$DATE.log
