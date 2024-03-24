# Check if DATE is provided as a command-line argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <date> (e.g., ./run.sh 12-25-13-10PM)"
    exit 1
fi

DATE="$1"

# Run the command with the provided date
# nohup python main.py --do-pre-train --pre-train-tasks mass --batch-size 16 --eval-batch-size 32 --cuda-visible-devices 0 --fp16 --model-name pre_train --pre-train-subset-ratio 0.001 --pre-train-parse-subset-ratio 0.01 --train-subset-ratio 0.001 --n-epoch 1 --n-epoch-pre-train 1 --remove-existing-saved-file > output-spt-code-2024-$DATE.log 2>&1 &

nohup python main.py --do-pre-train --pre-train-tasks mass --batch-size 16 --eval-batch-size 32 --cuda-visible-devices 0 --fp16 --model-name pre_train --pre-train-subset-ratio 0.1 --parse-subset-ratio 0.1 --train-subset-ratio 0.001 --n-epoch 1 --n-epoch-pre-train 1 --remove-existing-saved-file fine_tune:pre_train > output-spt-code-2024-$DATE.log 2>&1 &
