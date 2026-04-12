/*
** compact_db.c
**
** Manually run LSM compaction on a sqlite4_libsql database file.
** Use this after bulk inserts when LSM_DFLT_AUTOWORK=0.
**
** Build:
**   gcc -O2 compact_db.c -I. -Isrc/ -L. -lsqlite4 -lpthread -lm -o compact_db
**
** Usage:
**   ./compact_db /mnt/nvme0/mydb.db
*/
#include <stdio.h>
#include <stdlib.h>
#include "src/lsm.h"

int main(int argc, char **argv){
  lsm_db *pDb = 0;
  lsm_env *pEnv = lsm_default_env();
  int nWritten = 0;
  int nTotal = 0;
  int rc;

  if( argc < 2 ){
    fprintf(stderr, "Usage: %s <database_file>\n", argv[0]);
    return 1;
  }

  rc = lsm_new(pEnv, &pDb);
  if( rc != 0 ){
    fprintf(stderr, "lsm_new failed: %d\n", rc);
    return 1;
  }

  rc = lsm_open(pDb, argv[1]);
  if( rc != 0 ){
    fprintf(stderr, "lsm_open failed: %d\n", rc);
    lsm_close(pDb);
    return 1;
  }

  /* Show DB structure before compaction */
  {
    char *zInfo = 0;
    lsm_info(pDb, LSM_INFO_DB_STRUCTURE, &zInfo);
    fprintf(stderr, "Before: %s\n", zInfo ? zInfo : "(null)");
    lsm_free(pEnv, zInfo);
  }

  /* Phase 1: Merge with nMerge=4 to do bulk merge work */
  fprintf(stderr, "Compacting %s (phase 1: merge)\n", argv[1]);
  do {
    rc = lsm_work(pDb, 4, 4096, &nWritten);
    if( rc != 0 ){
      fprintf(stderr, "lsm_work failed: %d\n", rc);
      break;
    }
    nTotal += nWritten;
    if( nWritten > 0 ){
      fprintf(stderr, "  %d KB written so far\n", nTotal);
    }
  } while( nWritten > 0 );

  fprintf(stderr, "Phase 1 done. Total: %d KB written.\n", nTotal);

  /* Phase 2: Compact to single segment with nMerge=1.
  ** This is required before lsm_reclaim can safely move blocks,
  ** because the redirect array is shared across ALL segments. */
  if( rc == 0 ){
    fprintf(stderr, "Phase 2: compact to single segment\n");
    nTotal = 0;
    do {
      rc = lsm_work(pDb, 1, 4096, &nWritten);
      if( rc != 0 ){
        fprintf(stderr, "lsm_work(nMerge=1) failed: %d\n", rc);
        break;
      }
      nTotal += nWritten;
      if( nWritten > 0 ){
        fprintf(stderr, "  %d KB written so far\n", nTotal);
      }
    } while( nWritten > 0 );
    fprintf(stderr, "Phase 2 done. Total: %d KB written.\n", nTotal);
  }

  /* Show DB structure after merge */
  {
    char *zInfo = 0;
    lsm_info(pDb, LSM_INFO_DB_STRUCTURE, &zInfo);
    fprintf(stderr, "After merge: %s\n", zInfo ? zInfo : "(null)");
    lsm_free(pEnv, zInfo);
  }

  /* Reclaim free space */
  fprintf(stderr, "Reclaiming free space\n");
  nTotal = 0;
  do {
    rc = lsm_reclaim(pDb, 4096, &nWritten);
    if( rc != 0 ){
      fprintf(stderr, "lsm_reclaim failed: %d\n", rc);
      break;
    }
    nTotal += nWritten;
    if( nWritten > 0 ){
      fprintf(stderr, "  %d KB relocated so far\n", nTotal);
    }
  } while( nWritten > 0 );

  fprintf(stderr, "Reclaim done. Total: %d KB relocated.\n", nTotal);

  /* Show final DB structure */
  {
    char *zInfo = 0;
    lsm_info(pDb, LSM_INFO_DB_STRUCTURE, &zInfo);
    fprintf(stderr, "Final:  %s\n", zInfo ? zInfo : "(null)");
    lsm_free(pEnv, zInfo);
  }

  lsm_close(pDb);
  return 0;
}
