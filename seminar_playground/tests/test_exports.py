import json
import pandas as pd
from src.exports import csv_bytes, json_bytes, markdown_report, parity_png

def test_exports_are_real_files():
    assert b"rho" in csv_bytes(pd.DataFrame({"rho":[1]}))
    assert json.loads(json_bytes({"x":1})) == {"x":1}
    assert b"MLP Playground" in markdown_report({"epochs":1,"history":[]})
    assert parity_png(__import__("numpy").array([[0.,0.]]), __import__("numpy").array([[0.,0.]])).startswith(b"\x89PNG")
