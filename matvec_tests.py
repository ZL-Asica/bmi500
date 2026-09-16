import unittest

from matvec_multiply import dot_product, matvec_multiply


class MatvecTests(unittest.TestCase):
    def test_dot_product(self) -> None:
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)


    def test_dot_product_rejects_mismatched_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            dot_product([1], [1, 2])


    def test_matvec_multiply(self) -> None:
        self.assertEqual(matvec_multiply([[1, 2], [3, 4]], [5, 6]), [17, 39])


    def test_matvec_multiply_rejects_ragged_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "match the vector"):
            matvec_multiply([[1, 2], [3]], [4, 5])


    def test_empty_matrix(self) -> None:
        self.assertEqual(matvec_multiply([], [1, 2]), [])
        with self.assertRaises(TypeError):
            matvec_multiply([], [True])

    def test_empty_vectors_and_zero_columns(self) -> None:
        self.assertEqual(dot_product([], []), 0)
        self.assertEqual(matvec_multiply([[], [], []], []), [0, 0, 0])

    def test_rectangular_negative_and_float_inputs(self) -> None:
        self.assertEqual(matvec_multiply(((1, -2, 0.5), (-1, 0, 2)), (2, 3, 4)), [-2, 6])

    def test_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            matvec_multiply([[1, 2]], [1])

    def test_rejects_nonnumeric_and_nonfinite_inputs(self) -> None:
        for bad in ('1', None, True, 1j):
            with self.subTest(value=bad), self.assertRaises(TypeError):
                dot_product([bad], [1])
        for bad in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                dot_product([1], [bad])

        for bad in ('1', True, float('nan')):
            with self.subTest(matrix_value=bad), self.assertRaises((TypeError, ValueError)):
                matvec_multiply([[bad]], [1])
        for bad in ('1', True, float('nan')):
            with self.subTest(vector_value=bad), self.assertRaises((TypeError, ValueError)):
                matvec_multiply([[1]], [bad])

    def test_inputs_are_not_modified(self) -> None:
        matrix, vector = [[1, 2], [3, 4]], [5, 6]
        matvec_multiply(matrix, vector)
        self.assertEqual(matrix, [[1, 2], [3, 4]])
        self.assertEqual(vector, [5, 6])

    def test_identity_and_linearity(self) -> None:
        self.assertEqual(matvec_multiply([[1, 0], [0, 1]], [5, -3]), [5, -3])
        matrix = [[2, -1], [3, 4]]
        a, b = [1, 2], [3, -1]
        combined = matvec_multiply(matrix, [x + y for x, y in zip(a, b)])
        separate = [x + y for x, y in zip(matvec_multiply(matrix, a), matvec_multiply(matrix, b))]
        self.assertEqual(combined, separate)


if __name__ == "__main__":
    unittest.main()
