## [DBS]

1. 파일 및 디렉토리 설명
    
    `dataset/` : GloVe 200d , SIFT 128d 데이터 각각 insert/query/groundtruth 포함하는 디렉토리

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

1. 각 sqlite 실행코드 및 sqlite4의 compact_db 코드 컴파일
2. `recall_test.py`
    
    `LSM_CONFIG_AUTOWORK = 1` 로 빌드했을 때 

    ```
    python3 recall_test.py \
        --db-dir {db 저장할 디렉토리} \
        --k 10 \
        --auto-compact 1
    ```

    `LSM_CONFIG_AUTOWORK = 0` 으로 빌드했을 때 

    ```
    python3 recall_test.py \
        --db-dir {db 저장할 디렉토리} \
        --k 10 \
        --auto-compact 0
    ```

3. `incremental_test.py`

    `LSM_CONFIG_AUTOWORK = 1` 로 빌드했을 때 

    ```
    python3 incremental_test.py \
        --db-dir {db 저장할 디렉토리} \
        --k 10 \
        --auto-compact 1
    ```

    `LSM_CONFIG_AUTOWORK = 0` 으로 빌드했을 때 

    ```
    python3 incremental_test.py \
        --db-dir {db 저장할 디렉토리} \
        --k 10 \
        --auto-compact 0
    ```
