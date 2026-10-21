# ============================================================
# tests/create_sample_pdf.py  –  Generate a Sample Contract PDF
# ============================================================
# This script creates a small multi-page PDF that looks like a
# simple contract. It's used to test the PDF extraction module.
#
# Run with:
#     python tests/create_sample_pdf.py
#
# It produces:  data/sample_contract.pdf
# ============================================================

import pymupdf  # PyMuPDF


def create_sample_contract(output_path: str = "data/sample_contract.pdf"):
    """
    Build a 3-page sample contract PDF and save it to disk.

    Page 1 – Title & parties
    Page 2 – Terms & conditions
    Page 3 – Signature block
    """

    doc = pymupdf.open()  # create a brand-new empty PDF

    # ── Page 1: Title & Parties ─────────────────────────────
    page1 = doc.new_page(width=612, height=792)  # US Letter size in points
    page1.insert_text(
        (72, 80),                # (x, y) position in points
        "SERVICE AGREEMENT",
        fontsize=24,
        fontname="helv",         # Helvetica
    )
    page1.insert_text(
        (72, 130),
        (
            "This Service Agreement (the \"Agreement\") is entered into\n"
            "as of January 1, 2025, by and between:\n\n"
            "Party A: Acme Corporation\n"
            "  Address: 123 Business Ave, Suite 100, New York, NY 10001\n\n"
            "Party B: Global Services LLC\n"
            "  Address: 456 Commerce St, Floor 5, San Francisco, CA 94105\n\n"
            "WHEREAS, Party A desires to engage Party B to provide certain\n"
            "professional services, and Party B is willing to provide such\n"
            "services under the terms and conditions set forth herein."
        ),
        fontsize=11,
        fontname="helv",
    )

    # ── Page 2: Terms & Conditions ──────────────────────────
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text(
        (72, 80),
        "TERMS AND CONDITIONS",
        fontsize=18,
        fontname="helv",
    )
    page2.insert_text(
        (72, 120),
        (
            "1. SCOPE OF SERVICES\n"
            "Party B shall provide consulting services as described in\n"
            "Exhibit A attached hereto. Services shall include but not\n"
            "be limited to: software development, system integration,\n"
            "and technical advisory services.\n\n"
            "2. COMPENSATION\n"
            "Party A shall pay Party B a monthly fee of $15,000 (fifteen\n"
            "thousand dollars) for services rendered. Payment is due\n"
            "within 30 days of receipt of invoice.\n\n"
            "3. TERM AND TERMINATION\n"
            "This Agreement shall commence on the Effective Date and\n"
            "continue for a period of twelve (12) months, unless\n"
            "terminated earlier by either party with 30 days written\n"
            "notice.\n\n"
            "4. CONFIDENTIALITY\n"
            "Both parties agree to maintain the confidentiality of all\n"
            "proprietary information exchanged during the term of this\n"
            "Agreement. This obligation shall survive termination of\n"
            "this Agreement for a period of two (2) years.\n\n"
            "5. LIMITATION OF LIABILITY\n"
            "In no event shall either party be liable for any indirect,\n"
            "incidental, special, or consequential damages arising out\n"
            "of this Agreement, even if advised of the possibility of\n"
            "such damages."
        ),
        fontsize=11,
        fontname="helv",
    )

    # ── Page 3: Signatures ──────────────────────────────────
    page3 = doc.new_page(width=612, height=792)
    page3.insert_text(
        (72, 80),
        "SIGNATURES",
        fontsize=18,
        fontname="helv",
    )
    page3.insert_text(
        (72, 130),
        (
            "IN WITNESS WHEREOF, the parties hereto have executed this\n"
            "Agreement as of the date first written above.\n\n\n"
            "PARTY A: Acme Corporation\n\n"
            "Signature: _________________________\n"
            "Name:      John Smith\n"
            "Title:     Chief Executive Officer\n"
            "Date:      January 1, 2025\n\n\n"
            "PARTY B: Global Services LLC\n\n"
            "Signature: _________________________\n"
            "Name:      Jane Doe\n"
            "Title:     Managing Director\n"
            "Date:      January 1, 2025"
        ),
        fontsize=11,
        fontname="helv",
    )

    # Save the PDF
    doc.save(output_path)
    doc.close()
    print(f"[OK] Sample contract saved to: {output_path}")


if __name__ == "__main__":
    create_sample_contract()
