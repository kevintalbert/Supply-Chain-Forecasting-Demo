#!/usr/bin/env python3
"""Generate mock PDF supply agreement for CON-7781 (Turbine Oil) RAG demo."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        raise SystemExit("Install reportlab: pip install reportlab")

    contracts_dir = os.path.join(ROOT, "contracts")
    os.makedirs(contracts_dir, exist_ok=True)
    path = os.path.join(contracts_dir, "CON-7781_Turbine_Oil_Supply_Agreement.pdf")

    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    text = c.beginText(50, height - 72)
    text.setLeading(14)

    lines = [
        "MASTER SUPPLY AGREEMENT",
        "Contract ID: CON-7781",
        "Commodity: Turbine Oil (NSN 9150-01-123-4567)",
        "Parties: Liberty Industrial (Supplier) and Customer Logistics Command",
        "",
        "4.2 ESCALATION AND SURCHARGES.",
        "Pricing for lubricants under this Agreement tracks published commodity indices.",
        "If the Supplier's Energy Surcharge Index rises more than eight percent (8%)",
        "quarter-over-quarter, the Supplier may apply an Energy Surcharge to Unit Prices.",
        "Such surcharge adjustments may not exceed twelve percent (12%) per contract year",
        "and must be supported by the referenced index publication.",
        "",
        "4.3 EXPEDITED LOGISTICS.",
        "Orders shipped under SC-EXPEDITE code may include incremental freight pass-through",
        "charges identified separately on invoices issued within thirty (30) days of shipment.",
        "",
        "7.1 FORCE MAJEURE.",
        "Neither party is liable for delays due to events beyond reasonable control,",
        "including regional transportation disruptions affecting inbound feedstock.",
    ]
    for line in lines:
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
