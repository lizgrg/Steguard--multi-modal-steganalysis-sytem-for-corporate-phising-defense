import mysql.connector
import json
import datetime

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '7878',  # leave empty if no MySQL password set
    'database': 'stegshield',
    'port': 3306
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    try:
        conn = get_connection()
        print("✅ Database connected successfully!")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def save_scan(file_name, file_type, prediction, confidence, features=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO SCAN_JOB (user_id, file_name, file_type, status) VALUES (%s, %s, %s, %s)",
            (1, file_name, file_type.lower(), 'complete')
        )
        job_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO DETECTION_RESULT (job_id, model_id, prediction, confidence, modality, features_json) VALUES (%s, %s, %s, %s, %s, %s)",
            (job_id,
             1 if file_type.lower() in ['image', 'img'] else 2,
             prediction.lower(),
             confidence,
             file_type.lower(),
             json.dumps(features) if features else None)
        )
        cursor.execute(
            "INSERT INTO AUDIT_LOG (user_id, job_id, action, ip_address) VALUES (%s, %s, %s, %s)",
            (1, job_id, f'scan_{file_type.lower()}_{prediction.lower()}', '127.0.0.1')
        )
        conn.commit()
        cursor.close()
        conn.close()
        return job_id
    except Exception as e:
        print(f"Save scan error: {e}")
        return None

def get_all_scans():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                sj.job_id,
                sj.file_name,
                sj.file_type,
                sj.created_at,
                dr.prediction,
                dr.confidence
            FROM SCAN_JOB sj
            JOIN DETECTION_RESULT dr ON sj.job_id = dr.job_id
            ORDER BY sj.created_at DESC
            LIMIT 100
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"Get scans error: {e}")
        return []

def get_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as total FROM SCAN_JOB")
        total = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(*) as threats FROM DETECTION_RESULT WHERE prediction='stego'")
        threats = cursor.fetchone()['threats']
        cursor.close()
        conn.close()
        return {'total': total, 'threats': threats, 'clean': total - threats}
    except Exception as e:
        print(f"Get stats error: {e}")
        return {'total': 0, 'threats': 0, 'clean': 0}