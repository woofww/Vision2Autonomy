from pathlib import Path

import numpy as np
from PIL import Image

from vision2autonomy.cli import main


def test_cli_writes_grayscale_edge_map(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "nested" / "edges.png"
    source = np.zeros((24, 24), dtype=np.uint8)
    source[:, 12:] = 255
    Image.fromarray(source).save(input_path)

    exit_code = main([str(input_path), str(output_path), "--low", "20", "--high", "50"])

    assert exit_code == 0
    assert output_path.exists()
    result = np.asarray(Image.open(output_path))
    assert result.shape == source.shape
    assert np.count_nonzero(result) > 0

