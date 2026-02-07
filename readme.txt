python3 -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

export CUDA_VISIBLE_DEVICES=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=32
python3 task3.py







026-02-07 13:16:56.277178: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Loading TRAIN data...
Loading VAL data...
Loading TEST data...
Train slices: (3210, 128, 128, 1)
Val slices: (1116, 128, 128, 1)
Test slices: (1160, 128, 128, 1)
2026-02-07 13:17:46.160200: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1770450466.161950 2226030 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 18211 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB MIG 3g.20gb, pci bus id: 0000:47:00.0, compute capability: 8.0
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 126, 126, 32)        │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 61, 61, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 30, 30, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 57600)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │       7,372,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 3)                   │             387 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 7,392,131 (28.20 MB)
 Trainable params: 7,392,131 (28.20 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/8
2026-02-07 13:17:48.360038: I external/local_xla/xla/service/service.cc:163] XLA service 0x7fdbbc009bc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2026-02-07 13:17:48.360090: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (0): NVIDIA A100-SXM4-40GB MIG 3g.20gb, Compute Capability 8.0
2026-02-07 13:17:48.487155: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2026-02-07 13:17:48.847793: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91002
2026-02-07 13:17:49.074924: I external/local_xla/xla/service/gpu/autotuning/dot_search_space.cc:208] All configs were filtered out because none of them sufficiently match the hints. Maybe the hints set does not contain a good representative set of valid configs? Working around this by using the full hints set instead.
2026-02-07 13:17:49.749232: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_893', 8 bytes spill stores, 8 bytes spill loads

2026-02-07 13:17:50.050453: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_893', 492 bytes spill stores, 492 bytes spill loads

I0000 00:00:1770450483.924522 2296645 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
101/101 ━━━━━━━━━━━━━━━━━━━━ 22s 53ms/step - accuracy: 0.4960 - loss: 0.9992 - val_accuracy: 0.4014 - val_loss: 1.0826
Epoch 2/8
101/101 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.6056 - loss: 0.8508 - val_accuracy: 0.3916 - val_loss: 1.1466
Epoch 3/8
101/101 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.6822 - loss: 0.7289 - val_accuracy: 0.3961 - val_loss: 1.2203
Epoch 4/8
101/101 ━━━━━━━━━━━━━━━━━━━━ 5s 45ms/step - accuracy: 0.7586 - loss: 0.5882 - val_accuracy: 0.3943 - val_loss: 1.3930
Epoch 5/8
101/101 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.8171 - loss: 0.4778 - val_accuracy: 0.3746 - val_loss: 1.4660
Epoch 6/8
101/101 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.8713 - loss: 0.3668 - val_accuracy: 0.3737 - val_loss: 1.7129
Epoch 7/8
101/101 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.9097 - loss: 0.2799 - val_accuracy: 0.3754 - val_loss: 1.7508
Epoch 8/8
101/101 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.9377 - loss: 0.2122 - val_accuracy: 0.3629 - val_loss: 2.0707
37/37 ━━━━━━━━━━━━━━━━━━━━ 11s 41ms/step

===== FINAL EVALUATION (PATIENT LEVEL) =====
Balanced Accuracy : 0.354
Macro F1-Score    : 0.294
Precision (macro) : 0.296
Recall (macro)    : 0.354
AUC (OvR)         : 0.466

Saved Outputs:
 - confusion_matrix.png
 - accuracy_curve.png
 - loss_curve.png