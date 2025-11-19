import pytest

from QGA.LayerGA import buildLayer, randomLayer


def test_buildlayer_contains_rotations():
    # Build a known chromosome with X, R (Y), Z, I
    qc = buildLayer('XRYI', 4)
    instr_names = [instr[0].name for instr in qc.data]
    # Expect rx, ry and rz to appear (cx may or may not be present)
    assert any(name == 'rx' for name in instr_names), f"expected 'rx' in {instr_names}"
    assert any(name == 'ry' for name in instr_names), f"expected 'ry' in {instr_names}"
    assert any(name == 'rz' for name in instr_names), f"expected 'rz' in {instr_names}"


def test_randomlayer_length_and_format():
    for q in (1, 2, 4, 8):
        chrom = randomLayer(q)
        body = chrom.split('|')[0]
        assert len(body) == q
        # each char should be one of the supported tokens
        for ch in body:
            assert ch.upper() in ('R', 'X', 'Z', 'I')
