import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RAW_FOLDER = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_FOLDER = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_FOLDER, "cases.csv")

os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# ==========================
# CLEAN TEXT
# ==========================

def clean_text(text):

    # hapus watermark yang sering muncul
    patterns = [
        r'Direktori Putusan Mahkamah Agung Republik Indonesia',
        r'putusan\.mahkamahagung\.go\.id',
        r'Disclaimer.*?Halaman \d+',
    ]

    for p in patterns:
        text = re.sub(
            p,
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================
# NOMOR PERKARA
# ==========================

def get_nomor_perkara(text):

    match = re.search(
        r'Nomor\s+([\d\/A-Za-z\.\-]+)',
        text,
        re.IGNORECASE
    )

    return match.group(1) if match else ""


# ==========================
# PENGADILAN
# ==========================

def get_pengadilan(text):

    match = re.search(
        r'Pengadilan Negeri\s+([A-Za-z\s]+)',
        text
    )

    if match:
        return "PN " + match.group(1).strip()

    return ""


# ==========================
# PENGGUGAT
# ==========================

def get_penggugat(text):

    match = re.search(
        r'antara:\s*(.*?)\s*,.*?Penggugat',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return ""


# ==========================
# TERGUGAT
# ==========================

def get_tergugat(text):

    match = re.search(
        r'Lawan\s*:?\s*(.*?)\s*(?:Tergugat|Para Tergugat|Turut Tergugat)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        result = match.group(1)

        result = re.sub(r'\s+', ' ', result)

        return result.strip()

    return ""

# ==========================
# SHM
# ==========================

def get_objek_sengketa(text):

    match = re.search(
        r'Sertifikat Hak Milik.*?nomor\s*:?\s*([A-Za-z0-9\/\-]+)',
        text,
        re.IGNORECASE
    )

    if match:
        return f"SHM No.{match.group(1)}"

    return ""


# ==========================
# LUAS TANAH
# ==========================

def get_luas_tanah(text):

    match = re.search(
        r'seluas\s*([\d\.\,]+)\s*M2',
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return ""


# ==========================
# RINGKASAN FAKTA
# ==========================

def get_ringkasan_fakta(text):

    start = re.search(
        r'TENTANG DUDUK PERKARA',
        text,
        re.IGNORECASE
    )

    end = re.search(
        r'TENTANG PERTIMBANGAN HUKUM',
        text,
        re.IGNORECASE
    )

    if start and end:

        fakta = text[
            start.end():
            end.start()
        ]

        return fakta[:1500]

    return ""


# ==========================
# AMAR PUTUSAN
# ==========================

def get_amar_putusan(text):

    amar = ""

    match = re.search(
        r'MENGADILI:(.*?)(Demikian diputuskan)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        amar = match.group(1)

    return amar.strip()


# ==========================
# LABEL
# ==========================

def get_label(amar):

    amar_lower = amar.lower()

    if "dikabulkan sebagian" in amar_lower:
        return "Dikabulkan Sebagian"

    elif "dikabulkan" in amar_lower:
        return "Dikabulkan"

    elif "ditolak" in amar_lower:
        return "Ditolak"

    elif "tidak diterima" in amar_lower:
        return "Tidak Diterima"

    else:
        return "Lainnya"


# ==========================
# PROCESS
# ==========================

records = []

files = sorted(os.listdir(RAW_FOLDER))

for idx, filename in enumerate(files, start=1):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(
        RAW_FOLDER,
        filename
    )

    with open(
        filepath,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()

    text = clean_text(text)

    amar = get_amar_putusan(text)

    row = {

        "case_id": idx,

        "nomor_perkara":
        get_nomor_perkara(text),

        "pengadilan":
        get_pengadilan(text),

        "penggugat":
        get_penggugat(text),

        "tergugat":
        get_tergugat(text),

        "objek_sengketa":
        get_objek_sengketa(text),

        "luas_tanah":
        get_luas_tanah(text),

        "ringkasan_fakta":
        get_ringkasan_fakta(text),

        "amar_putusan":
        amar,

        "label":
        get_label(amar),

        "text_full":
        text
    }

    records.append(row)

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(df.head())

print()
print(f"Total Cases : {len(df)}")
print(f"Saved to : {OUTPUT_FILE}")