from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
import os

# Initialize VectorStore
vec = VectorStore()

# For Dev Only
create_table = False

data_root_files = "../data"
# List of CSV files
csv_files = [
    '1-0-เรียนล่วงหน้า.csv',
    '1-1-ช้างเผือก.csv',
    '1-1-รับนักกีฬาดีเด่น.csv',
    '1-1-นานาชาติและภาษาอังกฤษ.csv',
    '1-2-ช้างเผือก.csv',
    '1-2-โอลิมปิกวิชาการ.csv',
    '2-0-MOU.csv',
    '2-0-โควต้า30จังหวัด.csv',
    '2-0-เพชรนนทรี.csv',
    '2-0-ลูกพระพิรุณ.csv',
    '2-0-นานาชาติและภาษาอังกฤษ.csv',
    '2-0-ผู้มีความสามารถทางกีฬา.csv',
    '2-0-นักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์.csv',
    '3-0-Admission.csv',
]

# admission_round = "2"
# admission_program = "โควตานักกีฬา"

def prepare_record(row, admission_round, admission_program):
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
        f"สาขาวิชา: {row['สาขาวิชา']}\n"
        f"จำนวนรับ: {row['จำนวนรับ']}\n"
        f"เงื่อนไขขั้นต่ำ: {row['เงื่อนไขขั้นต่ำ']}\n"
        f"เกณฑ์การพิจารณา: {row['เกณฑ์การพิจารณา']}"
    )
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "major": row['สาขาวิชา'],
                "admission_round": admission_round,
                "admission_program": admission_program,
                "reference": row['แหล่งที่มา'],
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
    # get round and admission program    
    ro, sub_ro, program_t = csv_file.split("-")
    program_type = program_t[:-4]
    
    if (sub_ro == '1') or (sub_ro == '2'):
        ro += f"/{sub_ro}"
    
    csv_file_path = os.path.join(data_root_files, csv_file)
    # Read the CSV file
    df = pd.read_csv(csv_file_path, sep=",")
    df_eng = df[df['คณะ'] == "คณะวิศวกรรมศาสตร์"]
    
    # Apply the prepare_record function with the file name
    records_df = df_eng.apply(prepare_record, axis=1, admission_round=ro, admission_program=program_type)
    
    # Insert data into VectorStore
    vec.upsert(records_df)