"""Tiny synthetic corpora so the evaluations can run without any download.

``--smoke`` exercises exactly the same code path as a real run — sampling,
mapping, scoring, bootstrapping, JSON export, markdown rendering — on a handful
of hand-written provisions and contracts. It proves the plumbing works; it says
nothing at all about LexiChunk's real accuracy, and the numbers it produces
must never be quoted.
"""

from __future__ import annotations

from legal_rag_eval.evals.chunkers import Span
from legal_rag_eval.evals.cuad import Contract
from legal_rag_eval.evals.ledgar import LedgarSplit

#: (provision text, LEDGAR label) pairs covering a few unambiguous classes.
SMOKE_PROVISIONS: tuple[tuple[str, str], ...] = (
    (
        'As used herein, "Confidential Information" means any information '
        "disclosed by one party to the other party.",
        "Definitions",
    ),
    (
        "Each party shall hold in strict confidence all Confidential Information "
        "of the other party and shall not disclose any trade secret thereof.",
        "Confidentiality",
    ),
    (
        "The Borrower shall indemnify and hold harmless each Lender from and "
        "against any and all losses and damages.",
        "Indemnifications",
    ),
    (
        "This Agreement shall be governed by and construed in accordance with "
        "the laws of the State of New York.",
        "Governing Laws",
    ),
    (
        "This Agreement may be terminated by either party upon thirty days "
        "prior written notice of termination to the other party.",
        "Terminations",
    ),
    (
        "All notices hereunder shall be in writing and shall be deemed given "
        "when delivered personally or sent by certified mail.",
        "Notices",
    ),
    (
        "If any provision of this Agreement is held to be invalid or "
        "unenforceable, the remaining provisions shall remain in full force.",
        "Severability",
    ),
    (
        "This Agreement constitutes the entire agreement between the parties "
        "and supersedes all prior understandings.",
        "Entire Agreements",
    ),
    (
        "In no event shall either party's aggregate liability exceed the fees "
        "paid hereunder; in no event shall either party be liable for "
        "consequential damages.",
        "Remedies",
    ),
    (
        "Neither party may assign this Agreement without the prior written "
        "consent of the other party.",
        "Assignments",
    ),
    (
        "The Executive shall receive an annual base salary of $250,000, payable "
        "in accordance with the Company's regular payroll practices.",
        "Base Salary",
    ),
    (
        "The Company shall maintain commercial general liability insurance with "
        "coverage of not less than $1,000,000 per occurrence.",
        "Insurances",
    ),
)

#: A tiny two-clause contract used to smoke-test span containment.
_SMOKE_CONTRACT_A = """EXHIBIT 10.1

SERVICES AGREEMENT

Section 1. Definitions.

1.1 "Services" means the professional services described in Schedule A.

Section 2. Confidentiality.

2.1 Each party shall hold in confidence all Confidential Information disclosed
by the other party and shall not disclose it to any third party without prior
written consent.

Section 3. Indemnification.

3.1 Supplier shall indemnify and hold harmless Customer from any losses and
damages arising out of Supplier's negligence.

Section 4. Governing Law.

4.1 This Agreement shall be governed by the laws of the State of Delaware.

Section 5. Notices.

5.1 All notices shall be in writing and delivered to the addresses set forth
above.
"""

#: A contract with no recognisable headings, so the parser must fall back.
_SMOKE_CONTRACT_B = """This letter agreement confirms our understanding. The parties agree that all
confidential information exchanged shall remain confidential for a period of
three years. Either party may terminate this arrangement on written notice.
Any dispute shall be resolved by binding arbitration in New York.
"""


def smoke_ledgar_split(split: str) -> LedgarSplit:
    """Return the synthetic LEDGAR stand-in for any requested split."""
    return LedgarSplit(
        dataset_id="synthetic-smoke",
        config="ledgar",
        revision=None,
        texts=[text for text, _ in SMOKE_PROVISIONS],
        labels=[label for _, label in SMOKE_PROVISIONS],
    )


def smoke_cuad_contracts() -> tuple[str, str | None, list[Contract]]:
    """Return the synthetic CUAD stand-in corpus with verified gold spans."""
    contracts = []
    for name, text, phrases in (
        (
            "SMOKE_A",
            _SMOKE_CONTRACT_A,
            [
                "the professional services described in Schedule A",
                "hold in confidence all Confidential Information",
                "governed by the laws of the State of Delaware",
            ],
        ),
        (
            "SMOKE_B",
            _SMOKE_CONTRACT_B,
            [
                "confidential for a period of\nthree years",
                "resolved by binding arbitration in New York",
            ],
        ),
    ):
        spans = []
        for phrase in phrases:
            start = text.find(phrase)
            if start < 0:  # pragma: no cover - guards against edits to the fixtures
                msg = f"smoke fixture {name} does not contain {phrase!r}"
                raise AssertionError(msg)
            spans.append(Span(start, start + len(phrase)))
        contracts.append(
            Contract(
                contract_id=name,
                text=text,
                spans=tuple(spans),
                categories=("smoke",),
            )
        )
    return "synthetic-smoke", None, contracts
