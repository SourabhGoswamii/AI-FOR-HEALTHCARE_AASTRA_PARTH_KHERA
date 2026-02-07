python3 -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501




Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.

2026-02-07 06:02:23.158 Port 8501 is not available
vyakti7@scn69-mn:~/AI_Project/MRI_TASK2-3$ python3 test2.1.py
<frozen importlib._bootstrap_external>:1184: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
2026-02-07 06:02:43.428766: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Traceback (most recent call last):
  File "/nlsasfs/home/gpucbh/vyakti7/AI_Project/MRI_TASK2-3/test2.1.py", line 165, in <module>
    run_training()
  File "/nlsasfs/home/gpucbh/vyakti7/AI_Project/MRI_TASK2-3/test2.1.py", line 132, in run_training
    train_loader, val_loader = build_dataloaders()
  File "/nlsasfs/home/gpucbh/vyakti7/AI_Project/MRI_TASK2-3/test2.1.py", line 80, in build_dataloaders
    meta = pd.read_csv(META_FILE)
  File "/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
  File "/nlsasfs/home/gpucbh/vyakti7/.local/lib/python3.10/site-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: '/nlsasfs/home/gpucbh/vyakti7/Alzheimer_Project/data/metadata.csv'