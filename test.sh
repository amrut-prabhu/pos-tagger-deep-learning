#!/bin/bash
# File to store the accuracies of the POS tagger
TEST_RESULTS=test-results.txt
if [ ! -f $TEST_RESULTS ]
then
    touch $TEST_RESULTS
fi

COMMIT_HASH=$(git rev-parse HEAD)
echo "======$COMMIT_HASH======" >> $TEST_RESULTS

MODEL=model-file

# Build model
rm -f $MODEL
python buildtagger.py sents.train $MODEL

echo "=====================DONE=============="
echo "=====================DONE=============="
echo "=====================DONE=============="

# Run tests
rm -f sents.out
python runtagger.py sents.test $MODEL sents.out
ACCURACY=$(python eval.py sents.out sents.answer)
echo $ACCURACY
echo "sents $ACCURACY" >> $TEST_RESULTS
echo

rm -f 6.out
python runtagger.py 6.test $MODEL 6.out
ACCURACY=$(python eval.py 6.out 6.answer)
echo $ACCURACY
echo "Test6 $ACCURACY" >> $TEST_RESULTS
echo

rm -f 5.out
python runtagger.py 5.test $MODEL 5.out
ACCURACY=$(python eval.py 5.out 5.answer)
echo $ACCURACY
echo "Test5 $ACCURACY" >> $TEST_RESULTS
echo

rm -f 4.out
python runtagger.py 4.test $MODEL 4.out
ACCURACY=$(python eval.py 4.out 4.answer)
echo $ACCURACY
echo "Test4 $ACCURACY" >> $TEST_RESULTS
echo

rm -f 3.out
python runtagger.py 3.test $MODEL 3.out
ACCURACY=$(python eval.py 3.out 3.answer)
echo $ACCURACY
echo "Test3 $ACCURACY" >> $TEST_RESULTS
echo

rm -f 2.out
python runtagger.py 2.test $MODEL 2.out
ACCURACY=$(python eval.py 2.out 2.answer)
echo $ACCURACY
echo "Test2 $ACCURACY" >> $TEST_RESULTS
echo

rm -f 2a.out
python runtagger.py 2a.test $MODEL 2a.out
ACCURACY=$(python eval.py 2a.out 2a.answer)
echo $ACCURACY
echo "Test2a $ACCURACY" >> $TEST_RESULTS
echo

rm -f 2b.out
python runtagger.py 2b.test $MODEL 2b.out
ACCURACY=$(python eval.py 2b.out 2b.answer)
echo $ACCURACY
echo "Test2b $ACCURACY" >> $TEST_RESULTS
echo
