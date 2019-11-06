#!/bin/bash
# File to store the accuracies of the POS tagger
TEST_RESULTS=test-results.txt
if [ ! -f $TEST_RESULTS ]
then
    touch $TEST_RESULTS
fi

COMMIT_HASH=$(git rev-parse HEAD)
echo "======$COMMIT_HASH======" >> $TEST_RESULTS

rm -f model-file sents.out
python buildtagger.py sents.train model-file
python runtagger.py sents.test model-file sents.out
ACCURACY=$(python eval.py sents.out sents.answer)
echo $ACCURACY
echo "sents $ACCURACY" >> $TEST_RESULTS
echo

rm -f 2.out
python runtagger.py 2.test model-file 2.out
ACCURACY=$(python eval.py 2.out 2.answer)
echo $ACCURACY
echo "Test2 $ACCURACY" >> $TEST_RESULTS
echo

rm -f 3.out
python runtagger.py 3.test model-file 3.out
ACCURACY=$(python eval.py 3.out 3.answer)
echo $ACCURACY
echo "Test3 $ACCURACY" >> $TEST_RESULTS
echo

rm -f 4.out
python runtagger.py 4.test model-file 4.out
ACCURACY=$(python eval.py 4.out 4.answer)
echo $ACCURACY
echo "Test4 $ACCURACY" >> $TEST_RESULTS
echo
