python3 -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

CUDA_VISIBLE_DEVICES="" python3 task2.py


─────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ time_distributed (TimeDistributed)   │ (None, 16, 7, 7, 1024)      │       7,037,504 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ time_distributed_1 (TimeDistributed) │ (None, 16, 1024)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling1d             │ (None, 1024)                │               0 │
│ (GlobalAveragePooling1D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │         262,400 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 256)                 │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │             257 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 7,301,185 (27.85 MB)
 Trainable params: 263,169 (1.00 MB)
 Non-trainable params: 7,038,016 (26.85 MB)
Epoch 1/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 180s 1s/step - accuracy: 0.5909 - auc: 0.5000 - loss: 0.6929 - val_accuracy: 0.6000 - val_auc: 0.3148 - val_loss: 0.7574
Epoch 2/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 425ms/step - accuracy: 0.5909 - auc: 0.5085 - loss: 0.6919 - val_accuracy: 0.6000 - val_auc: 0.3241 - val_loss: 0.8178
Epoch 3/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 435ms/step - accuracy: 0.5909 - auc: 0.4573 - loss: 0.6906 - val_accuracy: 0.6000 - val_auc: 0.3148 - val_loss: 0.9162
Epoch 4/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 20s 444ms/step - accuracy: 0.5909 - auc: 0.4316 - loss: 0.6892 - val_accuracy: 0.6000 - val_auc: 0.3056 - val_loss: 1.0533
Epoch 5/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 433ms/step - accuracy: 0.5909 - auc: 0.4466 - loss: 0.6889 - val_accuracy: 0.6000 - val_auc: 0.3056 - val_loss: 1.2501
Epoch 6/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 425ms/step - accuracy: 0.5909 - auc: 0.4936 - loss: 0.6879 - val_accuracy: 0.6000 - val_auc: 0.3148 - val_loss: 1.5062
Epoch 7/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 426ms/step - accuracy: 0.5909 - auc: 0.4562 - loss: 0.6866 - val_accuracy: 0.6000 - val_auc: 0.3148 - val_loss: 1.8509
Epoch 8/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 443ms/step - accuracy: 0.5909 - auc: 0.4167 - loss: 0.6861 - val_accuracy: 0.6000 - val_auc: 0.2870 - val_loss: 2.2917
Epoch 9/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 428ms/step - accuracy: 0.5909 - auc: 0.4637 - loss: 0.6852 - val_accuracy: 0.6000 - val_auc: 0.5185 - val_loss: 2.8103
Epoch 10/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 435ms/step - accuracy: 0.5909 - auc: 0.5107 - loss: 0.6845 - val_accuracy: 0.6000 - val_auc: 0.5185 - val_loss: 3.4513
Epoch 11/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 432ms/step - accuracy: 0.5909 - auc: 0.3654 - loss: 0.6845 - val_accuracy: 0.6000 - val_auc: 0.4444 - val_loss: 4.2316
Epoch 12/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 428ms/step - accuracy: 0.5909 - auc: 0.3761 - loss: 0.6841 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 5.0843
Epoch 13/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 440ms/step - accuracy: 0.5909 - auc: 0.5481 - loss: 0.6824 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 6.0570
Epoch 14/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 441ms/step - accuracy: 0.5909 - auc: 0.4412 - loss: 0.6824 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 7.0980
Epoch 15/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 441ms/step - accuracy: 0.5909 - auc: 0.5000 - loss: 0.6823 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 8.1816
Epoch 16/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 432ms/step - accuracy: 0.5909 - auc: 0.5865 - loss: 0.6803 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 9.1772
Epoch 17/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 426ms/step - accuracy: 0.5909 - auc: 0.3686 - loss: 0.6818 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 10.0538
Epoch 18/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 428ms/step - accuracy: 0.5909 - auc: 0.5331 - loss: 0.6800 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 10.8477
Epoch 19/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 18s 419ms/step - accuracy: 0.5909 - auc: 0.4530 - loss: 0.6801 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 11.4165
Epoch 20/20
44/44 ━━━━━━━━━━━━━━━━━━━━ 19s 429ms/step - accuracy: 0.5909 - auc: 0.4145 - loss: 0.6794 - val_accuracy: 0.6000 - val_auc: 0.5000 - val_loss: 11.8647

FINAL TEST RESULTS (SUBJECT LEVEL)
1/1 ━━━━━━━━━━━━━━━━━━━━ 42s 42s/step
Balanced Accuracy: 0.5
AUC: 0.33333333333333337

Classification Report:
/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

           0       0.60      1.00      0.75         9
           1       0.00      0.00      0.00         6

    accuracy                           0.60        15
   macro avg       0.30      0.50      0.38        15
weighted avg       0.36      0.60      0.45        15

Confusion Matrix:
[[9 0]
 [6 0]]