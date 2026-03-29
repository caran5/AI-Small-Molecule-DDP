#!/usr/bin/env python3
"""Check ChEMBL database schema"""
import sqlite3
import tarfile
import tempfile
import os

db_path = 'src/data/chembl_34_sqlite.tar.gz'

with tempfile.TemporaryDirectory() as tmpdir:
    with tarfile.open(db_path, 'r:gz') as tar:
        tar.extractall(tmpdir)
    
    db_file = None
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            if f.endswith('.db'):
                db_file = os.path.join(root, f)
                break
        if db_file:
            break
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check molecule_dictionary columns
    cursor.execute("PRAGMA table_info(molecule_dictionary);")
    columns = cursor.fetchall()
    print("\nmolecule_dictionary columns:")
    for col in columns:
        print(f"  - {col[1]}")
    
    conn.close()
