# ============================================================
# tests/evaluation/synthetic_contract_generator.py
# ============================================================
# Generates a synthetic 3-page contract PDF with known clauses
# for quantitative RAG evaluation.
# ============================================================

import os
import pymupdf


def create_synthetic_evaluation_contract(output_path: str) -> str:
    """
    Generate a 3-page synthetic legal contract with known clauses
    for payment, termination, confidentiality, governing law,
    and out-of-domain evaluation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    doc = pymupdf.open()

    # ── Page 1: Title, Parties, and Scope ─────────────────────
    p1 = doc.new_page()
    text_p1 = """SYNTHETIC MASTER SERVICES AGREEMENT

This Synthetic Master Services Agreement (the "Agreement") is entered into as of October 15, 2025, by and between:

Party A: Apex Global Technologies Inc., a corporation organized under the laws of Delaware, having its principal place of business at 100 Tech Boulevard, Suite 400, Wilmington, DE 19801 ("Client"); and

Party B: Zenith Cloud Solutions LLC, a limited liability company organized under the laws of Delaware, having its principal place of business at 500 Enterprise Way, Floor 8, Dover, DE 19904 ("Service Provider").

RECITALS
WHEREAS, Client wishes to engage Service Provider to perform cloud database architecture migration, microservices deployment, infrastructure security audits, and continuous uptime monitoring services; and
WHEREAS, Service Provider possesses the technical expertise and personnel necessary to deliver said technical services in accordance with industry best practices.

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

1. SCOPE OF SERVICES
1.1 Deliverables: Service Provider shall deliver the cloud architecture migration as detailed in Exhibit A.
1.2 Standards: All services shall be performed in a professional, workmanlike manner in compliance with standard cloud security frameworks.
1.3 Acceptance Period: Client shall have ten (10) business days from receipt of deliverables to review and provide written acceptance.
"""
    p1.insert_textbox(pymupdf.Rect(54, 50, 550, 750), text_p1, fontsize=9.5)

    # ── Page 2: Compensation, Billing, and Retainer ───────────
    p2 = doc.new_page()
    text_p2 = """2. COMPENSATION AND PAYMENT TERMS

2.1 Monthly Base Service Fee:
Client shall pay Service Provider a fixed monthly base fee of $18,500 USD for cloud architecture and infrastructure management services rendered.

2.2 Invoicing Schedule & Net 30 Terms:
Service Provider shall submit itemized invoices on the first (1st) day of each calendar month. All undisputed invoice amounts are strictly due and payable within thirty (30) calendar days from the invoice date (Net 30 days).

2.3 Late Invoicing Interest:
Any undisputed amounts remaining unpaid after the Net 30 due date shall accrue late interest at the rate of 1.5% per month, or the maximum amount permitted by applicable law, whichever is less.

2.4 Security Retainer Deposit:
Upon mutual execution of this Agreement, Client shall provide an initial refundable security retainer deposit in the amount of $7,500 USD. The retainer shall be held in escrow and applied against any final month unpaid service balances.

2.5 Reimbursable Expenses:
Pre-approved out-of-pocket travel and third-party software licensing expenses exceeding $500 USD must be authorized in writing by Client prior to expenditure.
"""
    p2.insert_textbox(pymupdf.Rect(54, 50, 550, 750), text_p2, fontsize=9.5)

    # ── Page 3: Termination, Confidentiality, and Governing Law
    p3 = doc.new_page()
    text_p3 = """3. TERM, TERMINATION, CONFIDENTIALITY, AND GOVERNING LAW

3.1 Term:
This Agreement shall commence on October 15, 2025, and continue for an initial term of twelve (12) months.

3.2 Termination for Convenience:
Either party may terminate this Agreement without cause at any time by providing at least forty-five (45) calendar days prior written notice to the other party.

3.3 Termination for Cause:
Either party may terminate immediately if the other party materially breaches any term of this Agreement and fails to cure such breach within fifteen (15) days of receiving written notice specifying the breach.

3.4 Survival of Confidentiality Covenants:
Both parties acknowledge that Proprietary Information exchanged during the engagement constitutes sensitive business assets. All non-disclosure and confidentiality obligations set forth herein shall survive expiration or termination of this Agreement for a period of three (3) years.

3.5 Governing Law and Jurisdiction:
This Agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware, without giving effect to any principles of conflicts of law. The parties submit to the exclusive personal jurisdiction of the state and federal courts located in New Castle County, Delaware.
"""
    p3.insert_textbox(pymupdf.Rect(54, 50, 550, 750), text_p3, fontsize=9.5)

    doc.save(output_path)
    doc.close()
    return output_path


if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "synthetic_contract.pdf")
    create_synthetic_evaluation_contract(target)
    print(f"Generated synthetic evaluation contract at: {target}")
