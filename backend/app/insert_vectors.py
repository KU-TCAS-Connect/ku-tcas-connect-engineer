from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
import os

# Initialize VectorStore
vec = VectorStore()

# For Dev Only
create_table = False

# List of CSV files
# csv_files = ["../data/รอบ1-1ช้างเผือก.csv", "../data/รอบ1-2ช้างเผือก.csv", "../data/รอบ1เรียนล่วงหน้า.csv", "../data/รอบ2.csv", "../data/รอบ3.csv"]
# csv_files = ["../data-from-bell/รอบ1-2เกณฑ์โครงการช้างเผือก.csv", "../data-from-bell/รอบ2.csv"]
# csv_files = ["../data-from-bell/รอบ1เกณฑ์โครงการเรียนล่วงหน้า.csv"]
csv_files = ["../data-from-bell/รอบ2.csv"]

major = "วิศวกรรมเครื่องกล"
admission_round = "2"
admission_program = "โควตานักกีฬา"

def prepare_record(row, file_name):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Args:
        row: A row from the DataFrame.
        file_name: The name of the CSV file.
    """
    content = (
        f"รอบการคัดเลือก: {admission_round}\\n"
        f"โครงการ: {admission_program}\\n"
        f"สาขาวิชา: {row['สาขาวิชาที่ใช้เกณฑ์']}\n"
        f"จำนวนรับ: {row['จำนวนรับ']}\n"
        f"เงื่อนไขขั้นต่ำ: {row['เงื่อนไขขั้นต่ำ']}\n"
        f"เกณฑ์การพิจารณา: {row['เกณฑ์การพิจารณา']}"
    )
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "major": row['สาขาวิชาที่ใช้เกณฑ์'],
                "admission_round": admission_round,
                "admission_program": admission_program,
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


# Create tables and indexes once after processing all files
if create_table:
    vec.create_tables()
    vec.create_index()  # DiskAnnIndex

for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file, sep=",")
    
    # Apply the prepare_record function with the file name
    records_df = df.apply(prepare_record, axis=1, file_name=os.path.splitext(os.path.basename(csv_file))[0])
    
    # Insert data into VectorStore
    vec.upsert(records_df)