import csv
import sqlite3
import os
import argparse
from typing import List, Dict, Any, Optional, Tuple

def create_db_from_csv_files(mrconso_path: str, mrdef_path: str, relationships_path: str, db_path: str, test_mode: bool = False, limit: int = 200) -> None:
    """
    Create a SQLite database from UMLS CSV files.
    
    Args:
        mrconso_path: Path to the MRCONSO CSV file
        mrdef_path: Path to the MRDEF CSV file
        relationships_path: Path to the relationships (MRREL) CSV file
        db_path: Path to the output SQLite database file
        test_mode: Whether to run in test mode with limited data
        limit: Maximum number of entities to import in test mode
    """
    print(f"Creating SQLite database at {db_path}")
    if test_mode:
        print(f"Running in TEST MODE with limit of {limit} records per table")
    
    # Create/connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    create_tables(cursor)
    
    # Import data from CSV files
    cui_set = import_mrconso_data(cursor, mrconso_path, test_mode, limit)
    import_mrdef_data(cursor, mrdef_path, test_mode, limit, cui_set)
    import_relationships_data(cursor, relationships_path, test_mode, limit, cui_set)
    
    # Create indices for better query performance
    create_indices(cursor)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at {db_path}")


def create_tables(cursor: sqlite3.Cursor) -> None:
    """
    Create the necessary tables in the SQLite database.
    
    Args:
        cursor: SQLite cursor object
    """
    # Create MRCONSO table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mrconso (
        cui TEXT NOT NULL,
        lat TEXT,
        ts TEXT,
        lui TEXT,
        stt TEXT,
        sui TEXT,
        ispref TEXT,
        aui TEXT NOT NULL,
        saui TEXT,
        scui TEXT,
        sdui TEXT,
        sab TEXT,
        tty TEXT,
        code TEXT,
        str TEXT,
        srl INTEGER,
        suppress TEXT,
        cvf TEXT,
        PRIMARY KEY (cui, aui)
    )
    ''')
    
    # Create MRDEF table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mrdef (
        cui TEXT NOT NULL,
        aui TEXT NOT NULL,
        atui TEXT,
        satui TEXT,
        sab TEXT,
        def TEXT,
        suppress TEXT,
        cvf TEXT,
        PRIMARY KEY (cui, aui)
    )
    ''')
    
    # Create MRREL (relationships) table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mrrel (
        cui1 TEXT NOT NULL,
        aui1 TEXT NOT NULL,
        stype1 TEXT,
        rel TEXT,
        cui2 TEXT NOT NULL,
        aui2 TEXT NOT NULL,
        stype2 TEXT,
        rela TEXT,
        rui TEXT,
        srui TEXT,
        sab TEXT,
        sl TEXT,
        rg TEXT,
        dir TEXT,
        suppress TEXT,
        cvf TEXT,
        PRIMARY KEY (cui1, aui1, cui2, aui2, rela)
    )
    ''')
    
    print("Database tables created")


def import_mrconso_data(cursor: sqlite3.Cursor, file_path: str, test_mode: bool = False, limit: int = 200) -> set:
    """
    Import data from the MRCONSO CSV file into the SQLite database.
    
    Args:
        cursor: SQLite cursor object
        file_path: Path to the MRCONSO CSV file
        test_mode: Whether to run in test mode with limited data
        limit: Maximum number of records to import in test mode
        
    Returns:
        Set of CUI values imported (for use in related tables)
    """
    try:
        cui_set = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = []
            count = 0
            
            for row in reader:
                # Convert empty strings to None for numeric fields
                if 'SRL' in row and row['SRL'] == '':
                    row['SRL'] = None
                
                # Add CUI to set for tracking
                cui = row.get('CUI')
                if cui:
                    cui_set.add(cui)
                
                # Prepare row for insertion
                rows.append((
                    cui, row.get('LAT'), row.get('TS'), row.get('LUI'),
                    row.get('STT'), row.get('SUI'), row.get('ISPREF'), row.get('AUI'),
                    row.get('SAUI'), row.get('SCUI'), row.get('SDUI'), row.get('SAB'),
                    row.get('TTY'), row.get('CODE'), row.get('STR'), row.get('SRL'),
                    row.get('SUPPRESS'), row.get('CVF')
                ))
                
                count += 1
                # Stop after limit if in test mode
                if test_mode and count >= limit:
                    break
            
            # Insert all rows at once (more efficient)
            cursor.executemany('''
            INSERT OR REPLACE INTO mrconso (
                cui, lat, ts, lui, stt, sui, ispref, aui, saui, scui, 
                sdui, sab, tty, code, str, srl, suppress, cvf
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)
            
            print(f"Imported {len(rows)} records into MRCONSO table")
            return cui_set
            
    except Exception as e:
        print(f"Error importing MRCONSO data: {str(e)}")
        return set()


def import_mrdef_data(cursor: sqlite3.Cursor, file_path: str, test_mode: bool = False, limit: int = 200, cui_set: set = None) -> None:
    """
    Import data from the MRDEF CSV file into the SQLite database.
    
    Args:
        cursor: SQLite cursor object
        file_path: Path to the MRDEF CSV file
        test_mode: Whether to run in test mode with limited data
        limit: Maximum number of records to import in test mode
        cui_set: Set of CUI values to filter by (if provided)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = []
            count = 0
            
            for row in reader:
                # If we have a CUI set and this row's CUI is not in it, skip
                if cui_set and row.get('CUI') not in cui_set:
                    continue
                    
                # Prepare row for insertion
                rows.append((
                    row.get('CUI'), row.get('AUI'), row.get('ATUI'), row.get('SATUI'),
                    row.get('SAB'), row.get('DEF'), row.get('SUPPRESS'), row.get('CVF')
                ))
                
                count += 1
                # Stop after limit if in test mode
                if test_mode and count >= limit:
                    break
            
            # Insert all rows at once
            cursor.executemany('''
            INSERT OR REPLACE INTO mrdef (
                cui, aui, atui, satui, sab, def, suppress, cvf
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)
            
            print(f"Imported {len(rows)} records into MRDEF table")
    except Exception as e:
        print(f"Error importing MRDEF data: {str(e)}")


def import_relationships_data(cursor: sqlite3.Cursor, file_path: str, test_mode: bool = False, limit: int = 200, cui_set: set = None) -> None:
    """
    Import data from the relationships (MRREL) CSV file into the SQLite database.
    
    Args:
        cursor: SQLite cursor object
        file_path: Path to the relationships CSV file
        test_mode: Whether to run in test mode with limited data
        limit: Maximum number of records to import in test mode
        cui_set: Set of CUI values to filter by (if provided)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = []
            count = 0
            
            for row in reader:
                # If we have a CUI set, make sure both CUIs are in our set
                if cui_set and (row.get('CUI1') not in cui_set or row.get('CUI2') not in cui_set):
                    continue
                
                # Prepare row for insertion
                rows.append((
                    row.get('CUI1'), row.get('AUI1'), row.get('STYPE1'), row.get('REL'),
                    row.get('CUI2'), row.get('AUI2'), row.get('STYPE2'), row.get('RELA'),
                    row.get('RUI'), row.get('SRUI'), row.get('SAB'), row.get('SL'),
                    row.get('RG'), row.get('DIR'), row.get('SUPPRESS'), row.get('CVF')
                ))
                
                count += 1
                # Stop after limit if in test mode
                if test_mode and count >= limit:
                    break
            
            # Insert all rows at once
            cursor.executemany('''
            INSERT OR REPLACE INTO mrrel (
                cui1, aui1, stype1, rel, cui2, aui2, stype2, rela,
                rui, srui, sab, sl, rg, dir, suppress, cvf
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)
            
            print(f"Imported {len(rows)} records into MRREL table")
    except Exception as e:
        print(f"Error importing MRREL data: {str(e)}")


def create_indices(cursor: sqlite3.Cursor) -> None:
    """
    Create indices on the database tables for better query performance.
    
    Args:
        cursor: SQLite cursor object
    """
    # Create indices on MRCONSO table
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrconso_cui ON mrconso(cui)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrconso_aui ON mrconso(aui)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrconso_str ON mrconso(str)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrconso_lat ON mrconso(lat)')
    
    # Create indices on MRDEF table
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrdef_cui ON mrdef(cui)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrdef_aui ON mrdef(aui)')
    
    # Create indices on MRREL table
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrrel_cui1 ON mrrel(cui1)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrrel_aui1 ON mrrel(aui1)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrrel_cui2 ON mrrel(cui2)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrrel_aui2 ON mrrel(aui2)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mrrel_rela ON mrrel(rela)')
    
    print("Created indices for better query performance")


def query_db(db_path: str) -> None:
    """
    Execute some example queries on the database to test it.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\nExample Queries:")
    
    # 1. Get total counts
    cursor.execute("SELECT COUNT(*) FROM mrconso")
    print(f"Total MRCONSO records: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM mrdef")
    print(f"Total MRDEF records: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM mrrel")
    print(f"Total MRREL records: {cursor.fetchone()[0]}")
    
    # 2. Get English concepts only
    cursor.execute("SELECT COUNT(*) FROM mrconso WHERE lat = 'ENG'")
    print(f"English concepts: {cursor.fetchone()[0]}")
    
    # 3. Get concepts with definitions
    cursor.execute('''
    SELECT COUNT(DISTINCT m.cui, m.aui) 
    FROM mrconso m 
    JOIN mrdef d ON m.cui = d.cui AND m.aui = d.aui
    WHERE m.lat = 'ENG'
    ''')
    print(f"English concepts with definitions: {cursor.fetchone()[0]}")
    
    # 4. Sample concept with definition
    cursor.execute('''
    SELECT m.cui, m.aui, m.str, d.def
    FROM mrconso m
    JOIN mrdef d ON m.cui = d.cui AND m.aui = d.aui
    WHERE m.lat = 'ENG'
    LIMIT 1
    ''')
    result = cursor.fetchone()
    if result:
        print(f"\nSample concept:")
        print(f"CUI: {result[0]}, AUI: {result[1]}")
        print(f"STR: {result[2]}")
        print(f"Definition: {result[3]}")
    
    # 5. Sample relationship
    cursor.execute('''
    SELECT r.cui1, m1.str, r.rela, r.cui2, m2.str
    FROM mrrel r
    JOIN mrconso m1 ON r.cui1 = m1.cui AND r.aui1 = m1.aui
    JOIN mrconso m2 ON r.cui2 = m2.cui AND r.aui2 = m2.aui
    WHERE m1.lat = 'ENG' AND m2.lat = 'ENG'
    LIMIT 1
    ''')
    result = cursor.fetchone()
    if result:
        print(f"\nSample relationship:")
        print(f"Source: {result[0]} ({result[1]})")
        print(f"Relationship: {result[2] or 'RELATED_TO'}")
        print(f"Target: {result[3]} ({result[4]})")
    
    conn.close()


# JSON generation functionality has been removed


def main():
    parser = argparse.ArgumentParser(description='Create a SQLite database from UMLS CSV files')
    parser.add_argument('--mrconso', required=True, help='Path to the MRCONSO CSV file')
    parser.add_argument('--mrdef', required=True, help='Path to the MRDEF CSV file')
    parser.add_argument('--relationships', required=True, help='Path to the relationships CSV file')
    parser.add_argument('--db', default='tests/umls.db', help='Path to the output SQLite database file')
    parser.add_argument('--query', action='store_true', help='Run example queries after creating the database')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    parser.add_argument('--limit', type=int, default=200, help='Maximum number of records in test mode')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for file_path in [args.mrconso, args.mrdef, args.relationships]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Create the database
    create_db_from_csv_files(
        args.mrconso, 
        args.mrdef, 
        args.relationships, 
        args.db,
        test_mode=args.test,
        limit=args.limit
    )
    
    # Run example queries if requested
    if args.query:
        query_db(args.db)


if __name__ == "__main__":
    main()


# python create_database.py --mrconso ./level_2/diabetes_mrconso.csv --mrdef ./level_2/diabetes_mrdef.csv --relationships ./level_2/diabetes_relationships.csv --db ./level_2/umls.db --test --limit 200