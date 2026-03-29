"""
Inspect ChemBL database schema to find correct table and column names.
"""

import tarfile
import tempfile
import sqlite3
import os


def inspect_chembl_database():
    """Inspect the ChemBL database schema."""
    
    tar_path = '/Users/ceejayarana/diffusion_model/molecular_generation/src/data/chembl_34_sqlite.tar.gz'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Extracting database...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(tmpdir)
        
        # Find the database file
        db_file = None
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.endswith('.db'):
                    db_file = os.path.join(root, file)
                    break
        
        print(f"Found database: {db_file}\n")
        
        # Connect and inspect
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Tables in database:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Inspect first few tables
        print("\n" + "="*60)
        for table_name in ['COMPOUND_STRUCTURES', 'COMPOUNDS', 'MOLECULE_DICTIONARY', 'CHEMBL_ID_LOOKUP']:
            try:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                if columns:
                    print(f"\nTable: {table_name}")
                    print(f"Columns:")
                    for col in columns:
                        print(f"  - {col[1]} ({col[2]})")
                    
                    # Show first row
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                    row = cursor.fetchone()
                    if row:
                        print(f"Sample row: {row[:3]}")
            except:
                pass
        
        conn.close()


if __name__ == '__main__':
    inspect_chembl_database()
