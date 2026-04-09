## [DBS]

### 파일 및 디렉토리 설명
    
`dataset/` : 데이터 각각의 insert/query/groundtruth 포함

**데이터셋 용량문제로 인해 허깅페이스에 따로 업로드 했습니다.**

https://huggingface.co/datasets/jiwanyuk/recall_test_dataset/tree/main

다운로드 방법

1. 허깅페이스 라이브러리 설치
  
```
pip install huggingface-hub
```

2. ./dataset/ 디렉토리에 다운로드
```  
hf download jiwanyuk/recall_test_dataset --repo-type dataset --local-dir ./dataset
```
- 또는 특정 파일만 다운로드 (e.g. `insert100k_sift.sql`):
```
hf download jiwanyuk/recall_test_dataset insert100k_sift.sql --repo-type dataset --local-dir ./dataset
```

---

`sqlite4_lsm/` : 4kb, 16kb, 32kb, 64kb 페이지 사이즈 별 sqlite4 디렉토리 (컴파일 필요)

컴파일 방법 : 4kb, 16kb, 32kb, 64kb 각 디렉토리에서 
```
make -f Makefile.linux-gcc -B
```

추가로 각 디렉토리에서 
```
gcc -O2 compact_db.c -I. -Isrc/ -L. -lsqlite4 -lpthread -lm -o compact_db
```
컴파일 (수동으로 compaction 하는 코드)

---

`sqlite3_libsql` : 4kb, 16kb, 32kb, 64kb 페이지 사이즈 별 sqlite3 디렉토리 (컴파일 필요)

컴파일 방법 :
```
./configure
make -j
```

---

`recall_test.py` : glove, sift 데이터 각각 sqlite4, sqlite3 한번에 돌리는 코드

`incremental_test.py` : sqlite 데이터 각각에 대해 50% 빌드 후 10% 씩 incremental 하게 빌드하는 코드

### 테스트 코드 돌리는 법
0. numpy 설치된 파이썬 환경 준비
1. 각 sqlite 실행코드 및 sqlite4의 compact_db 코드 컴파일
2. `recall_test.py`
    
    `LSM_CONFIG_AUTOWORK = 1` 로 빌드했을 때 

    ```
    python3 recall_test.py --auto-compact 1
    ```

    `LSM_CONFIG_AUTOWORK = 0` 으로 빌드했을 때 

    ```
    python3 recall_test.py --auto-compact 0
    ```

    recall_test.py 안에 추가로 사용할 수 있는 옵션들 있으니 확인

3. `incremental_test.py`

    `LSM_CONFIG_AUTOWORK = 1` 로 빌드했을 때 

    ```
    python3 incremental_test.py --auto-compact 1
    ```

    `LSM_CONFIG_AUTOWORK = 0` 으로 빌드했을 때 

    ```
    python3 incremental_test.py --auto-compact 0
    ```

    incremental_test.py 안에 추가로 사용할 수 있는 옵션들 있으니 확인

### 예시 출력
```
Datasets: glove, sift
Configs:  lsm_4kb, lsm_16kb, lsm_32kb, lsm_64kb, sqlite3_4kb, sqlite3_16kb, sqlite3_32kb, sqlite3_64kb
DB dir:   /mnt/nvme0
Total runs: 16

######################################################################
  DATASET: glove
######################################################################
  Loaded 10000 groundtruth queries

============================================================
  Config: glove_lsm_4kb
  Shell:   /home/jiwan/sqlite4_lsm/4kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/4kb/compact_db
============================================================
  [1/4] Inserting...
        183.8s, 2674.0 MB
  [2/4] Compacting...
        5.8s, 2674.0 -> 1133.0 MB
        Final:  {0 {104040 104049 104049 10}} {2 {197156 32717 32717 134314}} {1 {157441 103943 103943 44295}} {1 {61953 30674 30674 27858}} {0 {3 4458 0 4456}}
  [3/4] Querying...
        25.93s (386 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.7104 (71.04%)

============================================================
  Config: glove_lsm_16kb
  Shell:   /home/jiwan/sqlite4_lsm/16kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/16kb/compact_db
============================================================
  [1/4] Inserting...
        179.9s, 2655.0 MB
  [2/4] Compacting...
        5.7s, 2655.0 -> 1139.0 MB
        Final:  {0 {7815 7818 7818 4}} {2 {51589 8391 8391 33987}} {1 {39233 25974 25974 11190}} {1 {15489 7814 7814 7046}} {0 {2 1116 0 1115}}
  [3/4] Querying...
        25.69s (389 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.7103 (71.03%)

============================================================
  Config: glove_lsm_32kb
  Shell:   /home/jiwan/sqlite4_lsm/32kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/32kb/compact_db
============================================================
  [1/4] Inserting...
        179.4s, 2663.0 MB
  [2/4] Compacting...
        5.9s, 2663.0 -> 1148.0 MB
        Final:  {0 {3984 3986 3986 3}} {2 {24490 4206 4206 17349}} {1 {19617 13143 13143 5687}} {1 {7745 3983 3983 3567}} {0 {2 558 0 557}}
  [3/4] Querying...
        25.59s (391 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.7109 (71.09%)

============================================================
  Config: glove_lsm_64kb
  Shell:   /home/jiwan/sqlite4_lsm/64kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/64kb/compact_db
============================================================
  [1/4] Inserting...
        179.5s, 2655.0 MB
  [2/4] Compacting...
        6.6s, 2655.0 -> 1166.0 MB
        Final:  {0 {6684 6684 0 1}} {2 {12884 2171 2171 8920}} {1 {9809 6673 6673 2913}} {1 {3873 2016 2016 1840}} {0 {2 280 0 279}}
  [3/4] Querying...
        25.61s (390 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.7092 (70.92%)

============================================================
  Config: glove_sqlite3_4kb
  Shell:   /home/jiwan/sqlite3_libsql/4kb/sqlite3
============================================================
  [1/3] Inserting...
        371.2s, 600.7 MB
  [2/3] Querying...
        31.77s (315 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.7095 (70.95%)

============================================================
  Config: glove_sqlite3_16kb
  Shell:   /home/jiwan/sqlite3_libsql/16kb/sqlite3
============================================================
  [1/3] Inserting...
        377.2s, 600.7 MB
  [2/3] Querying...
        33.69s (297 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.7109 (71.09%)

============================================================
  Config: glove_sqlite3_32kb
  Shell:   /home/jiwan/sqlite3_libsql/32kb/sqlite3
============================================================
  [1/3] Inserting...
        424.7s, 600.6 MB
  [2/3] Querying...
        36.82s (272 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.7100 (71.00%)

============================================================
  Config: glove_sqlite3_64kb
  Shell:   /home/jiwan/sqlite3_libsql/64kb/sqlite3
============================================================
  [1/3] Inserting...
        570.7s, 600.8 MB
  [2/3] Querying...
        46.65s (214 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.7104 (71.04%)

######################################################################
  DATASET: sift
######################################################################
  Loaded 10000 groundtruth queries

============================================================
  Config: sift_lsm_4kb
  Shell:   /home/jiwan/sqlite4_lsm/4kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/4kb/compact_db
============================================================
  [1/4] Inserting...
        153.7s, 2370.0 MB
  [2/4] Compacting...
        4.8s, 2370.0 -> 885.0 MB
        Final:  {0 {83545 83553 83553 9}} {2 {173313 24188 24188 96124}} {1 {132097 83472 83472 34576}} {1 {50689 21478 21478 19430}} {0 {3 3836 0 3834}}
  [3/4] Querying...
        22.44s (446 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.9868 (98.68%)

============================================================
  Config: sift_lsm_16kb
  Shell:   /home/jiwan/sqlite4_lsm/16kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/16kb/compact_db
============================================================
  [1/4] Inserting...
        151.1s, 2358.0 MB
  [2/4] Compacting...
        4.7s, 2358.0 -> 888.0 MB
        Final:  {0 {20923 20925 20925 3}} {2 {43265 6159 6159 24335}} {1 {32897 20898 20898 8738}} {1 {12609 5676 5676 4908}} {0 {2 957 0 956}}
  [3/4] Querying...
        22.46s (445 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.9870 (98.70%)

============================================================
  Config: sift_lsm_32kb
  Shell:   /home/jiwan/sqlite4_lsm/32kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/32kb/compact_db
============================================================
  [1/4] Inserting...
        150.7s, 2356.0 MB
  [2/4] Compacting...
        4.9s, 2356.0 -> 895.0 MB
        Final:  {0 {10549 10549 0 1}} {2 {21697 3125 3125 12405}} {1 {16417 10540 10540 4428}} {1 {6305 2827 2827 2475}} {0 {2 479 0 478}}
  [3/4] Querying...
        22.14s (452 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.9871 (98.72%)

============================================================
  Config: sift_lsm_64kb
  Shell:   /home/jiwan/sqlite4_lsm/64kb/sqlite4
  Compact: /home/jiwan/sqlite4_lsm/64kb/compact_db
============================================================
  [1/4] Inserting...
        151.1s, 2356.0 MB
  [2/4] Compacting...
        5.3s, 2356.0 -> 907.0 MB
        Final:  {0 {804 804 0 1}} {2 {9647 1619 1619 6341}} {1 {8209 5308 5308 2268}} {1 {3153 1469 1469 1277}} {0 {2 241 0 240}}
  [3/4] Querying...
        22.12s (452 q/s), 10000 queries returned
  [4/4] Computing recall@10...
        recall@10 = 0.9870 (98.70%)

============================================================
  Config: sift_sqlite3_4kb
  Shell:   /home/jiwan/sqlite3_libsql/4kb/sqlite3
============================================================
  [1/3] Inserting...
        335.1s, 448.8 MB
  [2/3] Querying...
        28.38s (352 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.9867 (98.67%)

============================================================
  Config: sift_sqlite3_16kb
  Shell:   /home/jiwan/sqlite3_libsql/16kb/sqlite3
============================================================
  [1/3] Inserting...
        367.1s, 442.6 MB
  [2/3] Querying...
        30.07s (333 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.9872 (98.72%)

============================================================
  Config: sift_sqlite3_32kb
  Shell:   /home/jiwan/sqlite3_libsql/32kb/sqlite3
============================================================
  [1/3] Inserting...
        510.5s, 399.2 MB
  [2/3] Querying...
        34.27s (292 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.9870 (98.70%)

============================================================
  Config: sift_sqlite3_64kb
  Shell:   /home/jiwan/sqlite3_libsql/64kb/sqlite3
============================================================
  [1/3] Inserting...
        706.3s, 398.9 MB
  [2/3] Querying...
        43.25s (231 q/s), 10000 queries returned
  [3/3] Computing recall@10...
        recall@10 = 0.9872 (98.72%)

================================================================================
  SUMMARY: glove (k=10)
================================================================================
          Config   Insert  Compact    Query   Before    After   Recall
                      (s)      (s)    (q/s)     (MB)     (MB)       @k
--------------------------------------------------------------------------------
         lsm_4kb    183.8      5.8      386   2674.0   1133.0   0.7104
        lsm_16kb    179.8      5.8      389   2655.0   1139.0   0.7103
        lsm_32kb    179.4      5.9      391   2663.0   1148.0   0.7109
        lsm_64kb    179.5      6.6      390   2655.0   1166.0   0.7092
     sqlite3_4kb    371.2      ---      315    600.7    600.7   0.7095
    sqlite3_16kb    377.2      ---      297    600.7    600.7   0.7109
    sqlite3_32kb    424.7      ---      272    600.6    600.6   0.7100
    sqlite3_64kb    570.7      ---      214    600.8    600.8   0.7104
================================================================================

================================================================================
  SUMMARY: sift (k=10)
================================================================================
          Config   Insert  Compact    Query   Before    After   Recall
                      (s)      (s)    (q/s)     (MB)     (MB)       @k
--------------------------------------------------------------------------------
         lsm_4kb    153.7      4.8      446   2370.0    885.0   0.9868
        lsm_16kb    151.1      4.7      445   2358.0    888.0   0.9870
        lsm_32kb    150.8      4.9      452   2356.0    895.0   0.9871
        lsm_64kb    151.1      5.3      452   2356.0    907.0   0.9870
     sqlite3_4kb    335.1      ---      352    448.8    448.8   0.9867
    sqlite3_16kb    367.1      ---      333    442.6    442.6   0.9872
    sqlite3_32kb    510.5      ---      292    399.2    399.2   0.9870
    sqlite3_64kb    706.3      ---      231    398.9    398.9   0.9872
================================================================================
```