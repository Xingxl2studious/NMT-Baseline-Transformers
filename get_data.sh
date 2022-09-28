pair=en-th
DATA_PATH=data

TOTAL_NUM=10000
VALID_NUM=5000
TEST_NUM=5000



if [ $pair == "en-th" ]; then
  mkdir $DATA_PATH/$pair
  echo "Download parallel data for English-Thai"
  # IIT Bombay English-Hindi Parallel Corpus
  wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-th.txt.zip -P $DATA_PATH/$pair --no-check-certificate
  unzip -u $DATA_PATH/$pair/download.php?f=OpenSubtitles2018%2Fen-th.txt.zip -d $DATA_PATH/$pair
fi



# lg=($(echo $pair | sed -e 's/\-/ /g'))
# echo ${lg[0]} ${lg[1]}
# paste $DATA_PATH/*.$pair.${lg[0]} $DATA_PATH/*.$pair.${lg[1]} -d' @@ ' > $DATA_PATH/$pair.all



# split into train / valid / test
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - $TOTAL_NUM));
    NVAL=$((NTRAIN + $VALID_NUM));
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN                   > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -$VALID_NUM  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -$TEST_NUM                 > $4;
}


for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  echo lg
  split_data $DATA_PATH/$pair/*.$pair.$lg $DATA_PATH/$pair/$pair.$lg.train $DATA_PATH/$pair/$pair.$lg.valid $DATA_PATH/$pair/$pair.$lg.test
done