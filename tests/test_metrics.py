from gogpx.metrics import avg_nearest_km, path_length_km


def test_path_length_zero():
    assert path_length_km([]) == 0.0
    assert path_length_km([(0.0, 0.0)]) == 0.0


def test_avg_nearest_empty():
    assert avg_nearest_km([], []) == float("inf")
    assert avg_nearest_km([(0.0, 0.0)], []) == float("inf")
