# export LD_LIBRARY_PATH=./baseLayerNormPlugin/:$LD_LIBRARY_PATH
# python builder.py -x ../../chapter_five/bert-base-uncased/model.onnx \
#                   -c ../../chapter_five/bert-base-uncased/ \
#                   -p ./bert_calibrator.cache \
#                   -i \
#                   -o ../../chapter_five/bert-base-uncased/model.plan -f | tee log.txt

python builder.py -x ../../chapter_five/bert-base-uncased/model.onnx \
                  -c ../../chapter_five/bert-base-uncased/ \
                  -o ../../chapter_five/bert-base-uncased/model.plan -f | tee log.txt
