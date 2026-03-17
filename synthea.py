import csv
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

BASE_DIR = Path(r"D:\Major Project\synthea")
CSV_DIR = BASE_DIR / "output" / "csv"
OUT_DIR = BASE_DIR / "output" / "bills"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOSPITALS = [
    ("Apollo Hospitals", "Hyderabad, Telangana"),
    ("Yashoda Hospitals", "Hyderabad, Telangana"),
    ("KIMS Hospitals", "Hyderabad, Telangana"),
    ("Fortis Hospital", "Bengaluru, Karnataka"),
    ("Manipal Hospital", "Bengaluru, Karnataka"),
    ("Narayana Health", "Bengaluru, Karnataka"),
    ("Aster Medcity", "Kochi, Kerala"),
    ("Amrita Institute of Medical Sciences", "Kochi, Kerala"),
    ("Lakeshore Hospital", "Kochi, Kerala"),
    ("Apollo Hospitals", "Chennai, Tamil Nadu"),
    ("MIOT International", "Chennai, Tamil Nadu"),
    ("Kauvery Hospital", "Chennai, Tamil Nadu"),
    ("KMCH", "Coimbatore, Tamil Nadu"),
    ("Ganga Hospital", "Coimbatore, Tamil Nadu"),
    ("KIMSHEALTH", "Thiruvananthapuram, Kerala"),
]

def read_csv(name):
    path = CSV_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def pick_patient(patients, idx):
    return patients[idx % len(patients)]

def pick_encounter(encounters, patient_id):
    for row in encounters:
        if row.get("PATIENT") == patient_id:
            return row
    return encounters[0]

def make_invoice_no(prefix, seed):
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{h}"

def currency(x):
    return f"INR {x:,.2f}"

def build_pdf(path, title, patient, encounter, rows, subtotal, tax_rate, hospital_name, hospital_city, invoice_date):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title)
    elements = []

    header = Paragraph(f"<b>{title}</b>", styles["Title"])
    elements.append(header)
    elements.append(Paragraph(hospital_name, styles["Normal"]))
    elements.append(Paragraph("GSTIN: 29ABCDE1234F1Z5", styles["Normal"]))
    elements.append(Paragraph(f"Address: {hospital_city}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    patient_name = f"{patient.get('FIRST', '')} {patient.get('LAST', '')}".strip()
    invoice_no = make_invoice_no(title.split()[0].upper(), patient.get("Id", "PAT") + invoice_date)
    encounter_class = encounter.get("ENCOUNTERCLASS", "").title()

    elements.append(Paragraph(f"<b>Invoice No:</b> {invoice_no}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date:</b> {invoice_date}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Patient:</b> {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Encounter:</b> {encounter_class}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Item", "Qty", "Unit Price", "Amount"]] + rows
    table = Table(table_data, colWidths=[250, 50, 90, 90])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3d5a80")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 12))

    tax = subtotal * tax_rate
    total = subtotal + tax

    elements.append(Paragraph(f"<b>Subtotal:</b> {currency(subtotal)}", styles["Normal"]))
    if tax_rate > 0:
        elements.append(Paragraph(f"<b>GST ({int(tax_rate * 100)}%):</b> {currency(tax)}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Total:</b> {currency(total)}", styles["Normal"]))

    doc.build(elements)

patients = read_csv("patients.csv")
encounters = read_csv("encounters.csv")
procedures = read_csv("procedures.csv")
medications = read_csv("medications.csv")

random.seed(42)

for h_idx, (hospital_name, city) in enumerate(HOSPITALS):
    hospital_dir = OUT_DIR / hospital_name.replace("/", "-")
    hospital_dir.mkdir(parents=True, exist_ok=True)

    for set_idx in range(1, 4):
        patient = pick_patient(patients, h_idx + set_idx)
        encounter = pick_encounter(encounters, patient.get("Id"))
        encounter_id = encounter.get("Id")
        proc_rows = [p for p in procedures if p.get("ENCOUNTER") == encounter_id]
        med_rows = [m for m in medications if m.get("ENCOUNTER") == encounter_id]

        invoice_date = (datetime.now() - timedelta(days=7 * (set_idx + h_idx))).strftime("%Y-%m-%d")

        hospital_items = [
            ("Room charges", 3 + set_idx, 2400.0 + random.randint(0, 400)),
            ("ICU charges", 1, 5800.0 + random.randint(0, 600)),
            ("Surgery package", 1, 17000.0 + random.randint(0, 2500)),
            ("Doctor fees", 1, 2200.0 + random.randint(0, 500)),
            ("Nursing charges", 1, 1100.0 + random.randint(0, 400)),
            ("Consumables", 1, 1400.0 + random.randint(0, 500)),
        ]

        if proc_rows:
            top_proc = proc_rows[0].get("DESCRIPTION", "Procedure")
            hospital_items[2] = (f"Surgery package - {top_proc}", 1, hospital_items[2][2])

        hospital_rows = [[name, str(qty), currency(unit), currency(qty * unit)] for name, qty, unit in hospital_items]
        hospital_subtotal = sum(qty * unit for _, qty, unit in hospital_items)

        pharmacy_items = []
        for m in med_rows[:6]:
            desc = m.get("DESCRIPTION") or "Medication"
            qty = 1
            unit = max(50.0, float(m.get("TOTALCOST", "0") or 0) / max(float(m.get("DISPENSES", "1") or 1), 1))
            pharmacy_items.append((desc, qty, unit))

        if not pharmacy_items:
            pharmacy_items = [
                ("Paracetamol 500mg", 10, 5.0),
                ("Amoxicillin 500mg", 10, 12.0),
                ("Pantoprazole 40mg", 10, 8.0),
                ("Saline 500ml", 2, 120.0),
                ("Syringe + IV set", 3, 60.0),
            ]

        pharmacy_rows = [[name, str(qty), currency(unit), currency(qty * unit)] for name, qty, unit in pharmacy_items]
        pharmacy_subtotal = sum(qty * unit for _, qty, unit in pharmacy_items)

        lab_items = []
        for p in proc_rows:
            desc = (p.get("DESCRIPTION") or "")
            if any(x in desc.lower() for x in ["lab", "blood", "x-ray", "ct", "mri", "scan", "ultrasound"]):
                lab_items.append((desc, 1, 600.0))

        if not lab_items:
            lab_items = [
                ("Complete Blood Count (CBC)", 1, 400.0),
                ("Blood Glucose", 1, 250.0),
                ("X-Ray Chest", 1, 700.0),
                ("Ultrasound Abdomen", 1, 1200.0),
            ]

        lab_rows = [[name, str(qty), currency(unit), currency(qty * unit)] for name, qty, unit in lab_items]
        lab_subtotal = sum(qty * unit for _, qty, unit in lab_items)

        prefix = f"Set{set_idx}_"
        build_pdf(hospital_dir / f"{prefix}Hospital_Bill.pdf", "Hospital Bill", patient, encounter, hospital_rows, hospital_subtotal, 0.0, hospital_name, city, invoice_date)
        build_pdf(hospital_dir / f"{prefix}Pharmacy_Bill.pdf", "Pharmacy Bill", patient, encounter, pharmacy_rows, pharmacy_subtotal, 0.05, hospital_name, city, invoice_date)
        build_pdf(hospital_dir / f"{prefix}Lab_Bill.pdf", "Lab Bill", patient, encounter, lab_rows, lab_subtotal, 0.0, hospital_name, city, invoice_date)

print(f"Created bill sets in: {OUT_DIR}")