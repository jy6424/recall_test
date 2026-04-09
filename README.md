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

