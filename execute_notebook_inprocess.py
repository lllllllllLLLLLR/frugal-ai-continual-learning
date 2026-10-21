from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO, BytesIO
import base64
import nbformat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

path = Path("StreamBench_Self_StreamICL_M1.ipynb")
nb = nbformat.read(path, as_version=4)
namespace = {"__name__": "__main__"}

def display(value):
    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False))
    else:
        print(value)

namespace["display"] = display

for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    cell.outputs = []
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)
        except Exception as exc:
            print(f"IN-PROCESS EXECUTION ERROR: {type(exc).__name__}: {exc}")
            raise
    if stdout.getvalue():
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stdout.getvalue()))
    if stderr.getvalue():
        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=stderr.getvalue()))
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": data}, metadata={}))
    plt.close("all")

nb.metadata["execution_note"] = "Executed in-process because the sandbox disallows Jupyter kernel sockets."
nbformat.write(nb, path)
print(f"Embedded outputs in {path}")
